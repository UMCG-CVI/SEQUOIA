"""
SEQUOIA - automated multiclass SEgmentation, QUantification, and visualizatiOn of the dIseased Aorta on hybrid PET/CT

@author: praaghgd
"""

import SimpleITK as sitk
from skimage import measure, morphology
import pymeshfix
from scipy.spatial.distance import cdist
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import meshio
import os
from opts import opt
from read_change_images import image_threshold, crop_image_to_label


def get_mesh_centerline(label_np, spacing):
    verts, faces, normals, values = measure.marching_cubes(label_np, spacing=spacing)
    meshfix = pymeshfix.MeshFix(verts, faces)
    meshfix.repair()
    verts = meshfix.v
    faces = meshfix.f
    cells = [("triangle", faces)]
    skel = morphology.skeletonize(label_np)
    return verts, cells, skel


def mesh_analysis(label, ct, calcium_threshold, output_path, pet=None, pet_threshold=None, pet_available=False):
    label_np = sitk.GetArrayFromImage(label)
    label_np[label_np>0] = 1
    label = sitk.GetImageFromArray(label_np)
    label.SetDirection(ct.GetDirection())
    label.SetSpacing(ct.GetSpacing())
    label.SetOrigin(ct.GetOrigin())

    if pet_available:
        pet, cropped_label = crop_image_to_label(pet, label)

    ct, label = crop_image_to_label(ct, label)
    label_np = sitk.GetArrayFromImage(label)
    label_np = np.transpose(label_np,(2,1,0))

    # sitk.WriteImage(ct, os.path.join(output_path,'ct_cropped.nii.gz'))
    # sitk.WriteImage(pet, os.path.join(output_path,'pet_cropped.nii.gz'))
    # sitk.WriteImage(label, os.path.join(output_path,'label_cropped.nii.gz'))
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(ct.GetPixelID())
    label = castFilter.Execute(label)
    mult_mask_ct = ct * label
    # sitk.WriteImage(mult_mask_ct, os.path.join(output_path,'mult_mask_ct.nii.gz'))
    ct_mask = image_threshold(mult_mask_ct, calcium_threshold)
    # sitk.WriteImage(ct_mask, os.path.join(output_path,'ct_mask.nii.gz'))
    ct_np = sitk.GetArrayFromImage(ct_mask)
    ct_np = np.transpose(ct_np,(2,1,0))

    if pet_available:
        castFilter.SetOutputPixelType(pet.GetPixelID())
        label = castFilter.Execute(label)
        mult_mask_pet = pet * label
        # sitk.WriteImage(mult_mask_pet, os.path.join(output_path,'mult_mask_pet.nii.gz'))
        pet_mask = image_threshold(mult_mask_pet, pet_threshold)
        # sitk.WriteImage(pet_mask, os.path.join(output_path,'pet_mask.nii.gz'))
        pet_np = sitk.GetArrayFromImage(pet_mask)
        pet_np = np.transpose(pet_np,(2,1,0))

    spacing = label.GetSpacing()

    verts, cells, skel = get_mesh_centerline(label_np, spacing)
    surf_coord = [[verts[i][0]/spacing[0], verts[i][1]/spacing[1], verts[i][2]/spacing[2]] for i in range(len(verts))]
    # skel_sitk = sitk.GetImageFromArray(skel)
    # skel_sitk.SetDirection(ct.GetDirection())
    # skel_sitk.SetSpacing(ct.GetSpacing())
    # skel_sitk.SetOrigin(ct.GetOrigin())
    # sitk.WriteImage(skel_sitk, os.path.join(output_path,'skel_sitk.nii.gz'))
    indices = np.where(skel==1)
    skel_coord = [[indices[0][i],indices[1][i],indices[2][i]] for i in range(len(indices[0]))]

    closest_skel_indices = np.argmin(cdist(surf_coord, skel_coord), axis=1)

    # Get the coordinates of the closest skeleton point for each surface point
    closest_skel_coords = np.take(skel_coord, closest_skel_indices, axis=0)

    # Create a RegularGridInterpolator for the CT matrix
    ct_interp = RegularGridInterpolator((range(ct_np.shape[0]), range(ct_np.shape[1]), range(ct_np.shape[2])), ct_np, bounds_error=False, fill_value=None)

    # Calculate the coordinates of all points on the lines connecting the surface points to the closest skeleton points
    line_coords = np.linspace(closest_skel_coords, surf_coord, num=100, axis=1)

    # Use the RegularGridInterpolator to get the CT values at all points on the lines
    line_ct_values = ct_interp(line_coords)

    # Sum the CT values for each line
    ct_values = line_ct_values.sum(axis=1)
    ct_values = [float(x) for x in ct_values]

    meshio.write_points_cells(
        os.path.join(output_path, 'ct_mesh.vtk'),
        verts,
        cells,
        point_data={'ct_values': ct_values}
    )

    if pet_available:
        pet_interp = RegularGridInterpolator((range(pet_np.shape[0]), range(pet_np.shape[1]), range(pet_np.shape[2])), pet_np, bounds_error=False, fill_value=None)
        line_pet_values = pet_interp(line_coords)
        pet_values = line_pet_values.sum(axis=1)
        pet_values = [float(x) for x in pet_values]
        meshio.write_points_cells(
            os.path.join(output_path, 'pet_mesh.vtk'),
            verts,
            cells,
            point_data={'pet_values': pet_values}
        )
        
        return ct_values, pet_values
    
    else:
        return ct_values