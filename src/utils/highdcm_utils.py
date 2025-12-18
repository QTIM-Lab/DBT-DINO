import time
from typing import Any

import highdicom as hd
import monai
import numpy as np
import pydicom
import torch
from dcmtrain.identifier import (
    DICOMInstanceIdentifier,
    DICOMSeriesIdentifier,
    RPSDICOMSeriesIdentifier,
)
from dicomweb_client.api import DICOMClient
from rps_client.client import RPSClient

# Overwrite the tolerance for the dot product of perpendicular vectors
hd.spatial._DOT_PRODUCT_PERPENDICULAR_TOLERANCE = 5e-1


def _read_dataset(
    p: DICOMSeriesIdentifier | DICOMInstanceIdentifier | RPSDICOMSeriesIdentifier | str | dict,
    client: DICOMClient | RPSClient | None = None,
):
    start_time = time.time()
    
    if isinstance(p, dict):
        if not isinstance(client, RPSClient):
            raise TypeError(
                "Transform must be configured with an RPSClient to "
                " retrieve data from dictionary."
            )
        # Create RPSDICOMSeriesIdentifier from dictionary data
        identifier = RPSDICOMSeriesIdentifier(
            mrn=p['mrn'],
            accession=p['accession'],
            series_instance_uid=p['series_instance_uid'],
            site=p['site']
        )
        series = client.retrieve_series(
            site=identifier.site,
            mrn=identifier.mrn,
            accession=identifier.accession,
            series_instance_uid=identifier.series_instance_uid,
        )
        if len(series) != 1:
            raise RuntimeError("Series must contain a single instance.")
        dataset = series[0]
    elif isinstance(p, DICOMInstanceIdentifier):
        if not isinstance(client, DICOMClient):
            raise TypeError(
                "Transform must be configured with a DICOMClient to "
                " retrieve DICOMInstanceIdentifiers."
            )
        iden = DICOMInstanceIdentifier.from_str(p)
        dataset = client.retrieve_instance(
            study_instance_uid=iden.study_instance_uid,
            series_instance_uid=iden.series_instance_uid,
            sop_instance_uid=iden.sop_instance_uid,
            media_types=(('application/dicom', '*'), )
        )
    elif isinstance(p, DICOMSeriesIdentifier):
        if not isinstance(client, DICOMClient):
            raise TypeError(
                "Transform must be configured with a DICOMClient to "
                " retrieve DICOMSeriesIdentifiers."
            )
        iden = DICOMSeriesIdentifier.from_str(p)
        series = client.retrieve_series(
            study_instance_uid=iden.study_instance_uid,
            series_instance_uid=iden.series_instance_uid,
        )
        if len(series) != 1:
            raise RuntimeError("Series must contain a single instance.")
        dataset = series[0]
    elif isinstance(p, RPSDICOMSeriesIdentifier):
        if not isinstance(client, RPSClient):
            raise TypeError(
                "Transform must be configured with an RPSClient to "
                " retrieve RPSDICOMSeriesIdentifier."
            )
        iden = RPSDICOMSeriesIdentifier.from_str(p)
        series = client.retrieve_series(
            site=iden.site,
            mrn=iden.mrn,
            accession=iden.accession,
            series_instance_uid=iden.series_instance_uid,
        )
        if len(series) != 1:
            raise RuntimeError("Series must contain a single instance.")
        dataset = series[0]
    else:
        dataset = pydicom.dcmread(p)

    load_file_time = time.time() - start_time

    #print(f"Loading file took {load_file_time:.4f} seconds")
    #logging.info(f"Loading file took {load_file_time:.4f} seconds")

    # Grab the view and laterality from the Series Description
    try: 
        view = str(dataset[(0x0008,0x103E)].value).split('Breast')[0].strip().replace('\x00', '').replace(' ', '')
    except:
        view = None

    return dataset, view


def _multiframe_dataset_to_metatensor(
    dataset: pydicom.Dataset,
    filename: str,
    is_dbt: bool = False,
    view: str | None = None,
) -> monai.data.MetaTensor:

    mf_image = hd.image.Image.from_dataset(dataset)

    # specify geometry
    start_time = time.time()
    try:
        vol = mf_image.get_volume(
            dtype=np.float32,
            apply_voi_transform=True,
        )
    except:
        print(f"Not a regularly spaced 3D volume: {filename}")
        #logging.info(f"Not a regularly spaced 3D volume: {filename}")
        vol = mf_image.get_volume(
            dtype=np.float32,
            apply_voi_transform=True,
            atol=1,
        )

    decomp_time = time.time() - start_time
    #print(f"Decompression took {decomp_time:.4f} seconds")
    #logging.info(f"Decompression took {decomp_time:.4f} seconds")

    vol = vol.ensure_handedness(handedness="RIGHT_HANDED", flip_axis=0) # TODO remove this for baseline ??

    num_vol_positions = vol.spatial_shape[0]
    last_patient_position = vol.map_indices_to_reference(
        np.array([[num_vol_positions - 1, 0, 0]])
    )

    image_unit_vectors = vol.unit_vectors()

    patient_frame_of_reference = (
        np.array([1, 0, 0]),
        np.array([0, 0, 1]),
        np.array([0, 1, 0])
    ) # TODO check this for all views

    # Compute dot products and perform flips if needed
    angles = []
    radians = []
    dot_products = []
    for idx, v in enumerate(image_unit_vectors):
        # Compute dot product of unit vector with patient axes
        dot_product = np.dot(v, patient_frame_of_reference[idx])
        dot_products.append(dot_product)
        # get angle in degrees
        angle = np.arccos(dot_product) * 180 / np.pi
        rad = np.arccos(dot_product)
        angles.append(angle)
        radians.append(rad)
    # Define a copy of the 3D volume to apply flips
    vol_ = vol
    affine = vol_.affine

    # Determine if view is MLO or CC for DBT images
    if is_dbt:
        affine[[0, 2]] = affine[[2, 0]]
        affine[[1, 2]] = affine[[2, 1]]
        # Parameter view is required
        if view is None:
            raise ValueError('View parameter is required for DBT images')

        if view == 'RCC':
            if dot_products[2] == -1.0:
                vol_ = vol.flip_spatial(axes=2)

        elif view == 'LCC':
            if dot_products[2] == 1.0:
                vol_ = vol.flip_spatial(axes=2)

        elif view == 'RXCCL':
            if dot_products[2] == -1.0:
                vol_ = vol.flip_spatial(axes=2)

        elif view == 'LXCCL':
            if dot_products[2] == 1.0:
                vol_ = vol.flip_spatial(axes=2)

        elif view == 'RMLO':
            if dot_products[0] < 0 and dot_products[1] > 0 and dot_products[2] == -1.0:
                # Flip along axes 2 and 1
                vol_ = vol.flip_spatial(axes=2).flip_spatial(axes=1)

        elif view == 'LMLO':
            if dot_products[0] < 0 and dot_products[1] > 0 and dot_products[2] == 1.0:
                # Flip along axes 2 and 1
                vol_ = vol.flip_spatial(axes=2).flip_spatial(axes=1)

        elif view == 'RML':
            if dot_products[0] < 0 and dot_products[1] > 0 and dot_products[2] == -1.0:
                # Flip along axes 2 and 1
                vol_ = vol.flip_spatial(axes=2).flip_spatial(axes=1)

        elif view == 'LML':
            if dot_products[0] < 0 and dot_products[1] > 0 and dot_products[2] == 1.0:
                # Flip along axes 2 and 1
                vol_ = vol.flip_spatial(axes=2).flip_spatial(axes=1)

        else:
            raise Warning(f'Received view: {view}. This is not a supported view for DBT images. Dot products: {dot_products}')

    meta_dict = {
        "affine": torch.from_numpy(affine),
        "original_affine": torch.from_numpy(vol.affine),
        "spacing": vol_.spacing,
        "filename_or_obj": filename,
        "lastImagePositionPatient": last_patient_position,
        "original_channel_dim": np.nan,
        "spatial_shape": vol_.spatial_shape,
        "space": monai.utils.SpaceKeys.LPS,
        "patient_orientation": str(mf_image['PatientOrientation'].value),
        "patient_frame_of_reference": patient_frame_of_reference,
        "unit_vectors": vol_.unit_vectors(),
    }


    return monai.data.MetaTensor(
        vol_.array.copy(),
        meta=meta_dict,
    )


class HighdicomMultiframeImageReader(monai.transforms.Transform):

    def __init__(self, client: DICOMClient | RPSClient | None = None, is_dbt: bool = False):
        self._client = client
        self._is_dbt = is_dbt

    def __call__(self, p: str):

        dataset, view = _read_dataset(p, client=self._client)
        return _multiframe_dataset_to_metatensor(dataset, p, is_dbt=self._is_dbt, view=view)


class HighdicomMultiframeImageReaderd(monai.transforms.Transform):

    """
    Class to read a multiframe DICOM image and return a MetaTensor.
    The parameter is_dbt should be set to True if the image is a multiframe DBT image.In this case, the keys should
    be the name of the view ('RCC', 'LCC', 'LMLO', 'RMLO') if view is not specified (default value is None). If another
    key that the view name is given, the view parameter should be a string corresponding to the view.
    """

    def __init__(self, keys: list[str], client: DICOMClient | RPSClient | None = None, is_dbt: bool = False, view: str | None = None, token: str | None = None):
        self.keys = keys
        self._client = client
        self._is_dbt = is_dbt
        self.view = view
        self.token = token

    def __call__(self, data: dict[str, Any]):
        start_time = time.time()

        data_out = data.copy()
        for k in self.keys:
            p = data[k]
            if self.token is not None:
                self._client = RPSClient(token=self.token)
            if self._is_dbt:
                dataset, view = _read_dataset(p, client=self._client)

                if view is None:
                    raise Warning(f'View not found for {p}')
                else:
                    view = view.upper()

                data_out[k] = _multiframe_dataset_to_metatensor(dataset, p, is_dbt=self._is_dbt, view=view)
            else:
                dataset, view = _read_dataset(p, client=self._client)
                data_out[k] = _multiframe_dataset_to_metatensor(dataset, p, is_dbt=self._is_dbt)

        total_time = time.time() - start_time  # End timing
        #print(f"HighdicomMultiframeImageReaderd applied in {total_time:.4f} seconds")
        #logging.info(f"HighdicomMultiframeImageReaderd applied in {total_time:.4f} seconds")

        return data_out
