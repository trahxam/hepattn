from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import uproot as up
import numpy as np


def get_parser():
    p = ArgumentParser(description="Convert root TTree files to binary parquet files")
    p.add_argument("-i", "--in_dir", dest="in_dir", type=str, required=True, help="Input directory containing root TTree files")
    p.add_argument("-o", "--out_dir", dest="out_dir", type=str, required=True, help="Where to save output binary files")
    p.add_argument("-t", dest="tree_name", required=False, default="hadronic_roi_tree", help="Name of TTree in .root files")
    p.add_argument("-d", "--debug", action="store_true", help="Print debug information")
    p.add_argument("--overwrite", action="store_true")
    return p


def preprocess(in_dir: str, out_dir: str, overwrite: bool):
    """Preprpocess root files into parquet files.

    Parameters
    ----------
    in_dir : str
        Directory of input root files
    out_dir : str
        Directory of where to save output parquet files
    overwrite : bool
        Whether to overwrite existing output files or not, by default false
    """

    # Iterate over all of the files ending in .root in the input directory
    for in_file in Path(in_dir).iterdir():
        if Path(in_file).suffix != ".root":
            continue

        out_file = Path(out_dir) / Path(in_file.stem).with_suffix(".parquet")
        
        # Only overwrite output files if overwrite flag is true
        if out_file.exists() and not overwrite:
            print(f"Skipping {in_file} as found existing output {out_file}")
            continue

        # The following dictionaries define mappings between the full name used in the ROOT
        # files / in Athena and a tuple which contains the aliased names that are used in this repo
        # along with the datatype of the field

        # Cluster fields that are common between pixel and SCT
        clus_fields = {
            "id": ("id", np.int32), # A unqiue integer ID for the cluster, going from 0 -> (num clusters in ROI - 1)
            "sihit_barcodes": ("bcodes", np.int32), # A list of barcodes of tracks that are associaed with this cluster
            "x": ("x", np.float32), 
            "y": ("y", np.float32),
            "z": ("z", np.float32),
            "layer": ("layer", np.int32),
            "bec": ("bec", np.float32),
            "charge": ("charge", np.float32),
            "module_global_x": ("mod_x", np.float32),
            "module_global_y": ("mod_y", np.float32),
            "module_global_z": ("mod_z", np.float32),
            "module_normal_x": ("mod_norm_x", np.float32),
            "module_normal_y": ("mod_norm_y", np.float32),
            "module_normal_z": ("mod_norm_z", np.float32),
            "module_local_x": ("mod_loc_x", np.float32),
            "module_local_y": ("mod_loc_y", np.float32),
            }
        
        # Pixel specific fields - note that it also has the common cluster fields
        pix_fields = clus_fields | {
            "NN_matrixOfCharge": ("chargemat", np.float32),
            "NN_vectorOfPitchesY":("pitches", np.float32),
            "NN_positions_indexX": ("sudo_loc_x", np.float32),
            "NN_positions_indexY": ("sudo_loc_y", np.float32),
            "NN_phi": ("sudo_loc_phi", np.float32),
            "NN_theta": ("sudo_loc_theta", np.float32),
            "LorentzShift": ("lshift", np.float32),
            "sihit_energyDeposits": ("energydep", np.float32),
            }
        
        # SCT specific fields
        sct_fields = clus_fields | {
            "side": ("side", np.float32),
            "SiWidth": ("width", np.float32),
            "rdo_strip": ("strip", np.float32),
            "rdo_groupsize": ("groupsize", np.float32),
            }
        
        # Track fields that are common betweeen all track collections
        trk_fields = {
            "id": ("id", np.int32), # A unqiue integer ID for the track, going from 0 -> (num tracks in ROI - 1)
            "barcode": ("bcode", np.int32), # A unique integer that identifies the particle corresponding to the track
            "to_cluster_ids": ("clus_ids", np.float32), # A list of cluster IDs that are associated with this track
            "pt": ("pt", np.float32),
            "eta": ("eta", np.float32),
            "phi": ("phi", np.float32),
            "d0": ("d0", np.float32),
            "z0": ("z0", np.float32),
            "vx": ("vx", np.float32),
            "vy": ("vy", np.float32),
            "vz": ("vz", np.float32),
            "charge": ("q", np.float32),
            "origin": ("origin", np.int32),
            }
        
        # Pseudotrack specific fields
        sudo_fields = trk_fields | {
            "BHadronPt": ("bhadpt", np.float32),
            "isComplete": ("complete", bool),
            "hasReco": ("hasreco", bool),
            "hasSiSP": ("hassisp", bool),
            }
        
        # Tracks produced by the standard Athena reconstruction chain
        reco_fields = trk_fields

        # SiSp tracks that are fed into the ambi
        sisp_fields = trk_fields
        
        # ROI level fields
        roi_fields = {
            "e": ("e", np.float32), # ROI energy
            "eta": ("eta", np.float32), # ROI axis eta
            "phi": ("phi", np.float32), # ROI axis phi
            "m": ("m", np.float32), # ROI mass
        }

        # Read the tree into memory
        print(f"\nLoading {in_file}\n")
        data = up.open(in_file)["hadronic_roi_tree"]

        data_out = {}

        if "NN_matrixOfCharge" in pix_fields.keys():
            # Sparsify the charge matrix to save memory
            pixel_mask = data["cluster_isPixel"].array()
            charges_dense = data["cluster_NN_matrixOfCharge"].array()[pixel_mask]
            idx = ak.broadcast_arrays(np.arange(49)[None, None, ...], data["cluster_id"].array()[pixel_mask])[0]
            data_out["pix_chargemat_val"] = charges_dense[charges_dense > 0]
            data_out["pix_chargemat_idx"] = idx[charges_dense > 0]
            del pix_fields["NN_matrixOfCharge"]

        for in_field, out_field_dtype in pix_fields.items():
            out_field, out_dtype = out_field_dtype
            data_out[f"pix_{out_field}"] = ak.values_astype(data[f"cluster_{in_field}"].array()[data["cluster_isPixel"].array()], out_dtype)

        for in_field, out_field_dtype in sct_fields.items():
            out_field, out_dtype = out_field_dtype
            data_out[f"sct_{out_field}"] = ak.values_astype(data[f"cluster_{in_field}"].array()[~data["cluster_isPixel"].array()], out_dtype)

        for in_field, out_field_dtype in sudo_fields.items():
            out_field, out_dtype = out_field_dtype
            data_out[f"sudo_{out_field}"] = ak.values_astype(data[f"pseudotracks_{in_field}"].array(), out_dtype)

        for in_field, out_field_dtype in reco_fields.items():
            out_field, out_dtype = out_field_dtype
            data_out[f"reco_{out_field}"] = ak.values_astype(data[f"recotracks_{in_field}"].array(), out_dtype)

        for in_field, out_field_dtype in sisp_fields.items():
            out_field, out_dtype = out_field_dtype
            data_out[f"sisp_{out_field}"] = ak.values_astype(data[f"sisptracks_{in_field}"].array(), out_dtype)

        for in_field, out_field_dtype in roi_fields.items():
            out_field, out_dtype = out_field_dtype
            data_out[f"roi_{out_field}"] = ak.values_astype(data[f"roi_{in_field}"].array(), out_dtype)

        data_out = ak.zip(data_out, depth_limit=1)

        print("-"*100 + "\nField name".ljust(32), "Type".ljust(32), "Bytes per ROI".ljust(32), "\n" + "-"*100)
        for field in data_out.fields:
            print(field.ljust(32), str(data_out[field].type).ljust(32), str(data_out[field].nbytes / (len(data_out))).ljust(32))

        size_per_roi_kb = data_out.nbytes / (len(data_out) * 1024)
        size_per_1m_roi_gb = 1000000 * size_per_roi_kb / (1024 * 1024)

        # Save to parquet
        ak.to_parquet(data_out, out_file)
        print(f"\nSaved {out_file}, size per ROI: {size_per_roi_kb:.3f}kB = {size_per_1m_roi_gb:.3f}GB per 1M ROIs\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    preprocess(args.in_dir, args.out_dir, args.overwrite)
