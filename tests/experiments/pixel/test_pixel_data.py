import numpy as np

from hepattn.experiments.pixel.data import PixelClusterDataset


def test_pixel_charge_to_valid_uses_strict_threshold():
    pixel_charge = np.array([0.0, 0.01, 0.05, 0.051, 1.2], dtype=np.float32)

    pixel_valid = PixelClusterDataset.pixel_charge_to_valid(pixel_charge, pixel_charge_thresh=0.05)

    np.testing.assert_array_equal(pixel_valid, np.array([False, False, False, True, True]))


def test_canonicalize_cluster_pitch_vector_reverses_raw_file_order():
    raw_cluster_pitch_vector = np.array([70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0], dtype=np.float32)

    cluster_pitch_vector = PixelClusterDataset.canonicalize_cluster_pitch_vector(raw_cluster_pitch_vector)

    np.testing.assert_array_equal(
        cluster_pitch_vector,
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32),
    )


def test_pixel_y_to_pitch_y_maps_center_row_and_clamps_edges():
    pixel_y = np.array([-10.0, -3.6, -2.6, -1.6, -0.4, 0.0, 0.8, 1.6, 2.6, 3.6], dtype=np.float32)
    cluster_pitch_vector = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32)

    pixel_pitch_y = PixelClusterDataset.pixel_y_to_pitch_y(pixel_y, cluster_pitch_vector)

    np.testing.assert_array_equal(pixel_pitch_y, np.array([10.0, 10.0, 20.0, 30.0, 40.0, 40.0, 50.0, 60.0, 70.0, 70.0], dtype=np.float32))


def test_pixel_x_to_real_uses_fixed_pitch():
    pixel_x = np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float32)

    pixel_x_real = PixelClusterDataset.pixel_x_to_real(pixel_x)

    np.testing.assert_allclose(pixel_x_real, np.array([-0.1, -0.025, 0.0, 0.075], dtype=np.float32))


def test_pixel_y_to_real_converts_centers_with_variable_row_pitch():
    pixel_y = np.array([-1.3, -0.3, 0.7, 1.7], dtype=np.float32)
    cluster_pitch_vector = np.array([30.0, 40.0, 50.0], dtype=np.float32)

    pixel_y_real = PixelClusterDataset.pixel_y_to_real(pixel_y, cluster_pitch_vector)

    np.testing.assert_allclose(pixel_y_real, np.array([-57.0, -12.0, 33.0, 88.0], dtype=np.float32))


def test_pixel_y_to_real_clamps_to_edge_pitch_outside_vector():
    pixel_y = np.array([-3.2, 2.8], dtype=np.float32)
    cluster_pitch_vector = np.array([20.0, 40.0, 60.0], dtype=np.float32)

    pixel_y_real = PixelClusterDataset.pixel_y_to_real(pixel_y, cluster_pitch_vector)

    np.testing.assert_allclose(pixel_y_real, np.array([-78.0, 162.0], dtype=np.float32))
