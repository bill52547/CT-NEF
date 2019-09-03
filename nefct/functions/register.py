# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: register.py
@date: 8/29/2019
@desc:
'''

from __future__ import print_function
from nefct import nef_class
from nefct.data import Image, Image2D, Image3D

import SimpleITK as sitk
import numpy as np


@nef_class
class Register:
    def __call__(self, fixed: Image, moved: Image) -> tuple:
        pass


class Register3D(Register):
    def __call__(self, fixed: Image3D, moved: Image3D) -> tuple:
        demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(50)
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(2.0)
        demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))
        fixed_image = sitk.GetImageFromArray(fixed.data)
        fixed_image.SetSpacing([fixed.size[i] / fixed.shape[i] for i in range(3)])
        moving_image = sitk.GetImageFromArray(moved.data)
        moving_image.SetSpacing([moved.size[i] / moved.shape[i] for i in range(3)])
        tx = multiscale_demons(registration_algorithm = demons_filter,
                               fixed_image = fixed_image,
                               moving_image = moving_image,
                               shrink_factors = [2, 1],
                               smoothing_sigmas = [2, 1])
        dvfs = sitk.GetArrayFromImage(tx.GetDisplacementField())
        return Image3D(dvfs[:, :, :, 0]), Image3D(dvfs[:, :, :, 1]), Image3D(dvfs[:, :, :, 2])


class Register2D(Register):
    def __call__(self, fixed: Image2D, moved: Image2D) -> tuple:
        demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(50)
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(2.0)
        demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))
        fixed_image = sitk.GetImageFromArray(fixed.data)
        fixed_image.SetSpacing([fixed.size[i] / fixed.shape[i] for i in range(2)])
        moving_image = sitk.GetImageFromArray(moved.data)
        moving_image.SetSpacing([moved.size[i] / moved.shape[i] for i in range(2)])
        tx = multiscale_demons(registration_algorithm = demons_filter,
                               fixed_image = fixed_image,
                               moving_image = moving_image,
                               shrink_factors = [2, 1],
                               smoothing_sigmas = [2, 1])
        dvfs = sitk.GetArrayFromImage(tx.GetDisplacementField())
        return Image3D(dvfs[:, :, 0]), Image3D(dvfs[:, :, 1])


def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors] * image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas] * image.GetDimension()

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(sf) + 0.5) for sf, sz in zip(shrink_factors, original_size)]
    new_spacing = [((original_sz - 1) * original_spc) / (new_sz - 1)
                   for original_sz, original_spc, new_sz in
                   zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())


def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None,
                      shrink_factors = None, smoothing_sigmas = None):
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(
                smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(
                smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[
                                                                           -1].GetSpacing(),
                                                                       fixed_images[
                                                                           -1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)

    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image,
                                                                    initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)


def iteration_callback(filter):
    print('\r{0}: {1:.2f}'.format(filter.GetElapsedIterations(), filter.GetMetric()), end = '')
