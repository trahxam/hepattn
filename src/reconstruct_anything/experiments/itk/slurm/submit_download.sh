#!/bin/bash

#SBATCH --job-name=itk-download
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/slurm-%j.%x.out

# Used for downloading raw ITk samples via rucio

# Move to repo root relative to this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"
echo "Working directory: ${PWD}"

# Set tmpdir
export TMPDIR=/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Where the data will be downloaded to
INPUT_DIR="/share/rcifdata/maxhart/data/itk/"

# Select which dataset to download
#DATASET_NAME="ATLAS-P2-RUN4-03-00-00_Rel.24_ttbar_uncorr_pu200_dumpv5_acorn2.0.0_acorn_data_reading_output_trainset"
DATASET_NAME="ATLAS-P2-RUN4-03-00-00_Rel.24_ttbar_uncorr_pu200_v9_acorn_data_reading_output_valset"
DATASET_TAG="user.avallier:${DATASET_NAME}"

cd $INPUT_DIR
echo "Moved dir, now in: ${PWD}"

# Setup atlas boilerplate
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup rucio

# Setup ssh key so we can activate the voms proxy
eval $(ssh-agent -s)
ssh-add /home/maxhart/.globus/userkey.pem

# Activate the proxy
voms-proxy-init -voms atlas

# Start the download
eval "rucio download ${DATASET_TAG}"

# Close the ssh agent
ssh-agent -k

# Remove the uneeded .pyg files
# eval "rm -rf ${INPUT_DIR}${DATASET_NAME}/*.pyg"

echo "Done!"
