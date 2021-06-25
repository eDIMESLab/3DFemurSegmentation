import itk
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import ctypes
#import matplotlib.pylab as plb
from scipy.ndimage.morphology import distance_transform_cdt
from scipy.ndimage.morphology import binary_erosion
import skimage.morphology
import os
import sys
#%%
# Make sure to link the built libraries
import fastDistMatrix
import GraphCutSupport
#%%
import warnings
import matplotlib.pylab as plb
from copy import deepcopy
from skopt import dump as skdump
from skopt import load as skload
from skopt.learning import GaussianProcessRegressor
from skopt.optimizer import base_minimize
from sklearn.utils import check_random_state
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import callbacks
#%%
def readNRRD(filename, ImageType):
  # Read series
  NRRDio = itk.NrrdImageIO.New()
  NRRDio.SetFileType( itk.ImageIOBase.ASCII )
  image_type = itk.Image[ImageType,3]
  reader = itk.ImageFileReader[image_type].New()
  reader.SetImageIO(NRRDio)
  reader.SetFileName(filename)
  reader.Update()
  imgs_seq = reader.GetOutput()
  metadatas = reader.GetMetaDataDictionary()
  # Store metadata as a list of dictionaries (more readable and less issues)
  meta_info = list(NRRDio.GetMetaDataDictionary().GetKeys())
  if 'NRRD_measurement frame' in meta_info:
    meta_info.remove('NRRD_measurement frame')
  # for tag in meta_info:
  #   try:
  #     print(metadatas[tag])
  #   except:
  #     print("ERROR:", tag)
  metadata = {tag:metadatas[tag] for tag in meta_info}
  return imgs_seq, metadata


# Dump 3D image into a 2D dicom image series with metadata adding option
def Volume3DToDicom(imgObj, MetadataObj = None, outdir = "", format_templ = "%03d.dcm"):
  format_templ = os.path.join(outdir, format_templ)
  entire_region = imgObj.GetLargestPossibleRegion()

  fngen = itk.NumericSeriesFileNames.New()
  fngen.SetSeriesFormat(format_templ)
  fngen.SetStartIndex( entire_region.GetIndex()[2] )
  fngen.SetEndIndex( entire_region.GetIndex()[2] + entire_region.GetSize()[2] - 1 )
  fngen.SetIncrementIndex(1)

  if not MetadataObj is None:
    metadata_array_copy = itk.vector.itkMetaDataDictionary()
    # I had to create a set of metadata to avoid pointers issues
    metadata_list_objs = [ itk.MetaDataDictionary() for metadata_list in MetadataObj ]
    for metadata_list, temp_metadata in zip(MetadataObj, metadata_list_objs):
      for k,v in metadata_list.items():
        temp_metadata[k] = v
      metadata_array_copy.append(temp_metadata)

  s,d = itk.template(imgObj)[1]
  dicom_io = itk.GDCMImageIO.New()
  writer_type = itk.Image[s,d]
  writer_otype = itk.Image[s,d-1]
  writer = itk.ImageSeriesWriter[writer_type, writer_otype].New()
  writer.SetInput(imgObj)
  writer.SetImageIO(dicom_io)
  writer.SetFileNames(fngen.GetFileNames())
  if not MetadataObj is None:
    writer.SetMetaDataDictionaryArray(metadata_array_copy)
  writer.Update()

# Read a dicom series in a directory and turn it into a 3D image
def dicomsTo3D(dirname, ImageType):
  seriesGenerator = itk.GDCMSeriesFileNames.New()
  seriesGenerator.SetUseSeriesDetails(True) # Use images metadata
  seriesGenerator.AddSeriesRestriction("0008|0021") # Series Date
  seriesGenerator.SetGlobalWarningDisplay(False); # disable warnings
  seriesGenerator.SetDirectory(dirname);

  # Get all indexes serieses and keep the longest series
  # (in doing so, we overcome the issue regarding the first CT sampling scan)
  seqIds = seriesGenerator.GetSeriesUIDs()
  UIDsFileNames = [ seriesGenerator.GetFileNames(seqId) for seqId in seqIds ]
  LargestDicomSetUID = np.argmax(list(map(len,UIDsFileNames)))
  LargestDicomSetFileNames = UIDsFileNames[LargestDicomSetUID]

  # Read series
  dicom_reader = itk.GDCMImageIO.New()
  image_type = itk.Image[ImageType,3]
  reader = itk.ImageSeriesReader[image_type].New()
  reader.SetImageIO(dicom_reader)
  reader.SetFileNames(LargestDicomSetFileNames)
  # Since CT acquisition is not fully orthogonal (gantry tilt)
  reader.ForceOrthogonalDirectionOff()
  reader.Update()
  imgs_seq = reader.GetOutput()
  metadataArray = reader.GetMetaDataDictionaryArray()
  # Store metadata as a list of dictionaries (more readable and less issues)
  dictionary_keys = list(filter(lambda x: not "ITK_" in x, dicom_reader.GetMetaDataDictionary().GetKeys()))
  metadata = [ {tag:fdcm[tag] for tag in dictionary_keys} for fdcm in metadataArray ]
  return imgs_seq, metadata


def thresholding(inputImage,
                 lowerThreshold = 25,
                 upperThreshold = 600):
  np_inputImage = itk.GetArrayFromImage(inputImage)
  np_inputImage = np.minimum( upperThreshold, np.maximum( np_inputImage, lowerThreshold) )
  result = itk.GetImageFromArray(np_inputImage)
  result.SetOrigin(inputImage.GetOrigin())
  result.SetSpacing(inputImage.GetSpacing())
  result.SetDirection(inputImage.GetDirection())
  return result


def castImage(imgObj, OutputType):
  s,d = itk.template(imgObj)[1]
  input_type = itk.Image[s,d]
  output_type = OutputType
  castObj = itk.CastImageFilter[input_type, output_type].New()
  castObj.SetInput(imgObj)
  castObj.Update()
  return castObj.GetOutput()


def computeRs(RsInputImg,
              # threshold = 1e-4,
              roi = None):
  eigenvalues_matrix = np.abs(RsInputImg)
  # eigenvalues_matrix[:,:,:,:3] = np.sort(eigenvalues_matrix[:,:,:,:3])
  det_image = np.sum(eigenvalues_matrix, axis=-1)
  if not roi is None:
    inside = roi!=0
    mean_norm = 1. / np.mean(det_image[inside])
    print("Mean norm = %f"% np.mean(det_image[inside]))
  else:
    mean_norm = 1. / np.mean(det_image)
    print("Mean norm = %f"% np.mean(det_image))
  Rnoise = det_image*mean_norm
  RsImage = np.empty(eigenvalues_matrix.shape[:-1] + (4,), dtype=float)
  al3_null = eigenvalues_matrix[:,:,:,2] != 0
  eigs = eigenvalues_matrix[al3_null,:3]
  tmp = 1./(eigs[:,1]*eigs[:,2])
  RsImage[al3_null, 0] = eigs[:,0]*tmp # Rtube
  tmp = 1. / eigs[:,2]
  RsImage[al3_null, 1] = eigs[:,1]*tmp # Rsheet
  tmp = 3. / det_image[al3_null]
  RsImage[al3_null, 2] = eigs[:,0]*tmp # Rblob
  RsImage[al3_null, 3] = Rnoise[al3_null] # Rnoise
  return RsImage, eigenvalues_matrix, al3_null


def computeSheetnessMeasure(SheetMeasInput,
                            roi = None,
                            alpha = 0.5,
                            beta = 0.5,
                            gamma = 0.5):
  if isinstance(SheetMeasInput, np.ndarray):
    sortedEigs = SheetMeasInput
    RsImg, EigsImg, NoNullEigs = computeRs(RsInputImg = SheetMeasInput, roi=roi)
  else:
    sortedEigs = itk.GetArrayFromImage(SheetMeasInput)
    # Sort them by abs (already done)
    # l1, l2, l3 = sortedEigs[:,:,:,0], sortedEigs[:,:,:,1], sortedEigs[:,:,:,2]
    # condA = np.abs(l1) > np.abs(l2)
    # l1[condA], l2[condA] = l2[condA], l1[condA]
    # condB = np.abs(l2) > np.abs(l3)
    # l2[condB], l3[condB] = l3[condB], l2[condB]
    # condC = np.abs(l1) > np.abs(l2)
    # l1[condC], l2[condC] = l2[condC], l1[condC]
    # sortedEigs[:,:,:,0], sortedEigs[:,:,:,1], sortedEigs[:,:,:,2] = l1, l2, l3
    RsImg, EigsImg, NoNullEigs = computeRs(RsInputImg = sortedEigs, roi=roi)
  SheetnessImage = np.empty(EigsImg.shape[:-1], dtype=float)
  SheetnessImage[NoNullEigs] = - np.sign( sortedEigs[NoNullEigs,2] )
  tmp = 1. / (beta*beta)
  SheetnessImage[NoNullEigs] *= np.exp(-RsImg[NoNullEigs,0] * RsImg[NoNullEigs,0] * tmp)
  tmp = 1. / (alpha*alpha)
  SheetnessImage[NoNullEigs] *= np.exp(-RsImg[NoNullEigs,1] * RsImg[NoNullEigs,1] * tmp)
  tmp = 1. / (gamma*gamma)
  SheetnessImage[NoNullEigs] *= np.exp(-RsImg[NoNullEigs,2] * RsImg[NoNullEigs,2] * tmp)
  # SheetnessImage *= EigsImg[:,:,:,2] ScaleObjectnessMeasureOff
  SheetnessImage[NoNullEigs] *= ( 1 - np.exp(-RsImg[NoNullEigs,3] * RsImg[NoNullEigs,3] * 4) )
  # SheetnessImage = itk.GetImageFromArray(SheetnessImage)
  return SheetnessImage





def computeEigenvalues(SmoothImg,
                       EigType,
                       D = 3):
  HessianImageType = type(SmoothImg)
  EigenValueArrayType = itk.FixedArray[EigType, D]
  EigenValueImageType = itk.Image[EigenValueArrayType, D]
  EigenAnalysisFilterType = itk.SymmetricEigenAnalysisImageFilter[HessianImageType]

  m_EigenAnalysisFilter = EigenAnalysisFilterType.New()
  m_EigenAnalysisFilter.SetDimension(D)
  m_EigenAnalysisFilter.SetInput(SmoothImg)
  m_EigenAnalysisFilter.Update()

  return m_EigenAnalysisFilter.GetOutput()


def GetEigenValues(M11, M12, M13, M22, M23, M33, roi=None):
  EigenValues = np.zeros(M11.shape + (4,))
  # Select roi
  if not roi is None:
    inside = roi!=0
    M11, M12, M13, M22, M23, M33 = M11[inside], M12[inside], M13[inside], M22[inside], M23[inside], M33[inside]
  a = -1.
  b = M11 + M22 + M33
  t12, t13, t23, s23 = M12*M12, M13*M13, M23*M23, M22*M33
  c = t12 + t13 + t23 - M11*(M22+M33) - s23
  d = M11*(s23 - t23) - M33*t12 + 2.*M12*M13*M23 - M22*t13
  x = ( (-3.*c) - (b*b) )
  tmpa = 1./3.
  x *= tmpa
  y = ((-2.*b*b*b) - (9.0*b*c) + (27.0*d/a))
  tmp = 1./27.
  y *= tmp
  z = y*y*0.25+x*x*x*tmp
  i = np.sqrt(y*y*0.25-z)
  j = -np.power(i, tmpa)
  k = np.arccos(-y/(2.0*i))
  m = np.cos(k*tmpa)
  n = np.sqrt(3.0)*np.sin(k*tmpa)
  p = -(b/(3.0*a))
  l1 = -2.0*j*m + p
  l2 = j*(m + n) + p
  l3 = j*(m - n) + p
  l1[np.isnan(l1)] = 0.
  l2[np.isnan(l2)] = 0.
  l3[np.isnan(l3)] = 0.
  condA = np.abs(l1) > np.abs(l2)
  l1[condA], l2[condA] = l2[condA], l1[condA]
  condB = np.abs(l2) > np.abs(l3)
  l2[condB], l3[condB] = l3[condB], l2[condB]
  condC = np.abs(l1) > np.abs(l2)
  l1[condC], l2[condC] = l2[condC], l1[condC]
  if not roi is None:
    EigenValues[inside,0] = l1
    EigenValues[inside,1] = l2
    EigenValues[inside,2] = l3
  else:
    EigenValues[:,:,:,0] = l1
    EigenValues[:,:,:,1] = l2
    EigenValues[:,:,:,2] = l3
  return EigenValues



def computeQuasiHessian(SmoothImg):
  SRImg = itk.GetArrayFromImage(SmoothImg)
  PaddedImg = np.pad(SRImg, 2,'edge')
  # Computing Values
  tmp = 2. * PaddedImg[2:-2,2:-2,2:-2]
  hxx = PaddedImg[2:-2,2:-2,:-4] - tmp + PaddedImg[2:-2,2:-2,4:] # ((-2,0,0) - 2*(0,0,0) + (2,0,0))/4.
  hyy = PaddedImg[2:-2,:-4,2:-2] - tmp + PaddedImg[2:-2,4:,2:-2] # ((0,-2,0) - 2*(0,0,0) + (0,2,0))/4.
  hzz = PaddedImg[:-4,2:-2,2:-2] - tmp + PaddedImg[4:,2:-2,2:-2] # ((0,0,-2) - 2*(0,0,0) + (0,0,2))/4.
  hxy = PaddedImg[2:-2,1:-3,1:-3] - PaddedImg[2:-2,1:-3,3:-1] - PaddedImg[2:-2,3:-1,1:-3] + PaddedImg[2:-2,3:-1, 3:-1] # ((-1,-1,0) - (1,-1,0) - (-1,1,0) + (1,1,0))/4.
  hxz = PaddedImg[1:-3,2:-2,1:-3] - PaddedImg[1:-3,2:-2,3:-1] - PaddedImg[3:-1,2:-2,1:-3] + PaddedImg[3:-1,2:-2, 3:-1] # ((-1,-1,0) - (1,-1,0) - (-1,1,0) + (1,1,0))/4.
  hyz = PaddedImg[1:-3,1:-3,2:-2] - PaddedImg[1:-3,3:-1,2:-2] - PaddedImg[3:-1,1:-3,2:-2] + PaddedImg[3:-1,3:-1,2:-2] # ((-1,-1,0) - (1,-1,0) - (-1,1,0) + (1,1,0))/4.
  # Division over 4
  hxx *= 0.25
  hyy *= 0.25
  hzz *= 0.25
  hxy *= 0.25
  hxz *= 0.25
  hyz *= 0.25
  return hxx, hxy, hxz, hyy, hyz, hzz



def computeHessian(HessInput,
                   sigma,
                   HRGImageType):
  HessianFilterType = itk.HessianRecursiveGaussianImageFilter[HRGImageType]
  HessianObj = HessianFilterType.New()
  HessianObj.SetSigma(sigma)
  HessianObj.SetInput(HessInput)
  HessianObj.Update()
  return HessianObj.GetOutput()


def SmoothingRecursive(SRInput,
                       sigma,
                       SRImageType):
  SRFilterType = itk.SmoothingRecursiveGaussianImageFilter[SRImageType, SRImageType]
  SRObj = SRFilterType.New()
  SRObj.SetInput(SRInput)
  SRObj.SetSigma( sigma )
  SRObj.Update()
  output_image = SRObj.GetOutput()
  output_image.DisconnectPipeline()
  return output_image


def singlescaleSheetness(singleScaleInput,
                         scale,
                         SmoothingImageType,
                         roi = None,
                         alpha = 0.5,
                         beta = 0.5,
                         gamma = 0.5):
  print("Computing single-scale sheetness, sigma=%4.2f"% scale)
  SmoothImg = SmoothingRecursive(SRInput = singleScaleInput,
                                 sigma = scale,
                                 SRImageType = SmoothingImageType)
  HessianMatrices = computeQuasiHessian(SmoothImg)
  EigenImg = GetEigenValues(*HessianMatrices, roi)
  # Alternative version:
  # SmoothImg = computeHessian(HessInput = singleScaleInput,
  #                            sigma = scale,
  #                            HRGImageType = SmoothingImageType)
  # EigenImg = computeEigenvalues(SmoothImg = SmoothImg,
  #                               EigType = itk.ctype('float'),
  #                               D = 3)
  SheetnessImg = computeSheetnessMeasure(SheetMeasInput = EigenImg,
                                         roi = roi,
                                         alpha = alpha,
                                         beta = beta,
                                         gamma = gamma)

  if not roi is None:
    SheetnessImg[roi==0] = 0.
  return SheetnessImg





def multiscaleSheetness(multiScaleInput,
                        scales,
                        SmoothingImageType,
                        roi = None,
                        alpha = 0.5,
                        beta = 0.5,
                        gamma = 0.5):
  if not roi is None:
    roi = itk.GetArrayFromImage(roi)
  multiscaleSheetness = singlescaleSheetness(singleScaleInput = multiScaleInput,
                                             scale = scales[0],
                                             SmoothingImageType = SmoothingImageType,
                                             roi = roi,
                                             alpha = alpha,
                                             beta = beta,
                                             gamma = gamma)

  if len(scales) > 1:
    for scale in scales[1:]:
      singleScaleSheetness  = singlescaleSheetness(multiScaleInput,
                                                   scale = scale,
                                                   SmoothingImageType = SmoothingImageType,
                                                   roi = roi,
                                                   alpha = alpha,
                                                   beta = beta,
                                                   gamma = gamma)
      refinement = abs(singleScaleSheetness) > abs(multiscaleSheetness)
      multiscaleSheetness[refinement] = singleScaleSheetness[refinement]
  multiscaleSheetness = itk.GetImageFromArray(multiscaleSheetness.astype(np.float32))
  multiscaleSheetness.SetOrigin(multiScaleInput.GetOrigin())
  multiscaleSheetness.SetSpacing(multiScaleInput.GetSpacing())
  multiscaleSheetness.SetDirection(multiScaleInput.GetDirection())
  return multiscaleSheetness


def binaryThresholding(inputImage,
                       lowerThreshold,
                       upperThreshold,
                       outputImageType = None,
                       insideValue = 1,
                       outsideValue = 0):
  # Old version:
  # s,d = itk.template(inputImage)[1]
  # input_type = itk.Image[s,d]
  # output_type = input_type if outputImageType is None else itk.Image[outputImageType,d]
  # thresholder = itk.BinaryThresholdImageFilter[input_type, output_type].New()
  # thresholder.SetInput(inputImage)
  # thresholder.SetLowerThreshold( lowerThreshold )
  # thresholder.SetUpperThreshold( upperThreshold )
  # thresholder.SetInsideValue(insideValue)
  # thresholder.SetOutsideValue(outsideValue)
  # thresholder.Update()
  # return thresholder.GetOutput()
  values = itk.GetArrayFromImage(inputImage)
  cond = (values>=lowerThreshold) & (values<=upperThreshold)
  values[ cond ] = insideValue
  values[ np.logical_not(cond) ] = outsideValue
  result = itk.GetImageFromArray(values)
  result.SetOrigin(inputImage.GetOrigin())
  result.SetSpacing(inputImage.GetSpacing())
  result.SetDirection(inputImage.GetDirection())
  if not outputImageType is None:
    s,d = itk.template(inputImage)[1]
    output_type = itk.Image[outputImageType,d]
    result = castImage(result, OutputType=output_type)
  return result



def ConnectedComponents(inputImage,
                        outputImageType = None):
  s,d = itk.template(inputImage)[1]
  input_type = itk.Image[s,d]
  output_type = input_type if outputImageType is None else itk.Image[outputImageType,d]
  CC = itk.ConnectedComponentImageFilter[input_type, output_type].New()
  CC.SetInput(inputImage)
  CC.Update()
  return CC.GetOutput()


def RelabelComponents(inputImage,
                      outputImageType = None):
  # relabel = itk.RelabelComponentImageFilter[input_type, output_type].New()
  # relabel.SetInput(inputImage)
  # relabel.Update()
  # return relabel.GetOutput()
  label_field = itk.GetArrayFromImage(inputImage)
  offset = 1
  max_label = int(label_field.max()) # Ensure max_label is an integer
  labels, labels_counts= np.unique(label_field,return_counts=True)
  labels=labels[np.argsort(labels_counts)[::-1]]
  labels0 = labels[labels != 0]
  new_max_label = offset - 1 + len(labels0)
  new_labels0 = np.arange(offset, new_max_label + 1)
  output_type = label_field.dtype
  required_type = np.min_scalar_type(new_max_label)
  if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
      output_type = required_type
  forward_map = np.zeros(max_label + 1, dtype=output_type)
  forward_map[labels0] = new_labels0
  inverse_map = np.zeros(new_max_label + 1, dtype=output_type)
  inverse_map[offset:] = labels0
  relabeled = forward_map[label_field]
  result = itk.GetImageFromArray(relabeled)
  result.SetOrigin(inputImage.GetOrigin())
  result.SetSpacing(inputImage.GetSpacing())
  result.SetDirection(inputImage.GetDirection())
  if not outputImageType is None:
    s,d = itk.template(inputImage)[1]
    output_type = itk.Image[outputImageType,d]
    result = castImage(result, OutputType=output_type)
  return result


def Gaussian(GaussInput,
             sigma,
             outputImageType = None):
  s,d = itk.template(GaussInput)[1]
  input_type = itk.Image[s,d]
  output_type = input_type if outputImageType is None else itk.Image[outputImageType,d]
  OperationObj = itk.DiscreteGaussianImageFilter[input_type, output_type].New()
  OperationObj.SetInput(GaussInput)
  OperationObj.SetVariance(sigma)
  OperationObj.Update()
  return OperationObj.GetOutput()


def substract(ImgA,
              ImgB):
  s,d = itk.template(ImgA)[1]
  assert s,d == itk.template(ImgB)[1]
  input_type = itk.Image[s,d]
  Result = itk.SubtractImageFilter[input_type, input_type, input_type].New()
  Result.SetInput1(ImgA)
  Result.SetInput2(ImgB)
  Result.Update()
  return Result.GetOutput()


def add(ImgA,
        ImgB):
  s,d = itk.template(ImgA)[1]
  assert s,d == itk.template(ImgB)[1]
  input_type = itk.Image[s,d]
  Result = itk.AddImageFilter[input_type, input_type, input_type].New()
  Result.SetInput1(ImgA)
  Result.SetInput2(ImgB)
  Result.Update()
  return Result.GetOutput()


def linearTransform(Img,
                    scale,
                    shift,
                    outputImageType = None):
  s,d = itk.template(Img)[1]
  input_type = itk.Image[s,d]
  output_type = input_type if outputImageType is None else itk.Image[outputImageType,d]
  Result = itk.ShiftScaleImageFilter[input_type, output_type].New()
  Result.SetInput(Img)
  Result.SetScale(scale)
  Result.SetShift(shift)
  Result.Update()
  return Result.GetOutput()


# Old:
# def DistanceTransform(ChamferInput):
#   distanceMap = np.zeros(ChamferInput.shape)
#   _infinityDistance = np.sum(ChamferInput.shape) + 1
#   distanceMap[ChamferInput == 0] = _infinityDistance
#   distanceMapPad = np.pad(distanceMap, 1, mode='constant', constant_values=(_infinityDistance, _infinityDistance))
#   distanceMap = fastDistMatrix.ManhattanChamferDistance(distanceMapPad, distanceMap.shape)
#   distanceMap = distanceMap[1:-1, 1:-1, 1:-1].copy()
#   return distanceMap

def DistanceTransform(ChamferInput):
  distanceMap = np.zeros(ChamferInput.shape)
  _infinityDistance = np.sum(ChamferInput.shape) + 1
  distanceMap[ChamferInput == 0] = _infinityDistance
  distanceMapPad = np.pad(distanceMap, 1, mode='constant', constant_values=(_infinityDistance, _infinityDistance))
  distanceMapPad_flat = distanceMapPad.flatten("C").astype(np.int_)
  distanceMap = fastDistMatrix.ComputeChamferDistance(distanceMapPad_flat,
                                                      len(distanceMapPad_flat),
                                                      distanceMap.shape[0],
                                                      distanceMap.shape[1],
                                                      distanceMap.shape[2]
                                                      )
  distanceMap = distanceMap.reshape(distanceMapPad.shape)
  distanceMap = distanceMap[1:-1, 1:-1, 1:-1].copy()
  return distanceMap

# depth  = distanceMap.shape[0] + 1
# height = distanceMap.shape[1] + 1
# width  = distanceMap.shape[2] + 1
# area   = (width+1)*(height+1)
# im = distanceMapPad_flat
# for k in range(1,depth):
#   trail_k = k*area
#   for j in range(1,height):
#     trail_j = j*(width+1)
#     trail_jk = trail_j + trail_k
#     for i in range(1, width):
#       pos = i + trail_jk
#       pos_i = pos + 1
#       pos_j = pos + (width+1)
#       pos_k = pos + area
#       # pixel = std :: min({im[pos],
#       #                     im[pos_i] + weight,
#       #                     im[pos_j] + weight,
#       #                     im[pos_k] + weight});
#       assert im[pos] == distanceMapPad[k,j,i]
#       assert im[pos_i] == distanceMapPad[k,j,i+1]
#       assert im[pos_j] == distanceMapPad[k,j+1,i]
#       assert im[pos_k] == distanceMapPad[k+1,j,i]

#%%
########################
# Segmentation section #
########################

# SheetnessBasedSmoothCost_compute:
def SheetnessBasedSmoothCost(pixelLeft,
                             pixelRight,
                             shtnLeft,
                             shtnRight,
                             COST_AMPLIFIER = 1000,
                             alpha = 5.0):
  cond = (pixelLeft > -1) & (pixelRight > -1)
  smoothCostFromCenter = np.ones(pixelLeft[cond].shape)
  smoothCostToCenter   = np.ones(pixelLeft[cond].shape)
  dSheet = abs(shtnLeft[cond] - shtnRight[cond])
  # From Center
  cond_b = shtnLeft[cond] >= shtnRight[cond]
  smoothCostFromCenter[cond_b] = np.exp(- 5. * dSheet[cond_b])
  smoothCostFromCenter = (smoothCostFromCenter * COST_AMPLIFIER * alpha + 1).astype(np.int32)
  # To Center
  cond_b = np.logical_not(cond_b)
  smoothCostToCenter[cond_b] = np.exp(- 5. * dSheet[cond_b])
  smoothCostToCenter = (smoothCostToCenter * COST_AMPLIFIER * alpha + 1).astype(np.int32)
  return (pixelLeft[cond], pixelRight[cond]), smoothCostFromCenter, smoothCostToCenter


def Segmentation(imgObj,
                 softEst,
                 sht,
                 ROI,
                 COST_AMPLIFIER = 1000,
                 alpha = 5.0):
  # assignIdsToPixels
  intensity  = itk.GetArrayFromImage(imgObj)
  softTissue = itk.GetArrayFromImage(softEst)
  sheetness  = itk.GetArrayFromImage(sht)
  _pixelIdImage = np.zeros(intensity.shape)
  roi = itk.GetArrayFromImage(ROI)
  _pixelIdImage[roi==0] = -1
  _totalPixelsInROI = np.sum(roi!=0)
  _pixelIdImage[roi!=0] = range(_totalPixelsInROI)
  # SheetnessBasedDataCost_compute for initializeDataCosts prep
  # - BONE, 0
  dataCostSink = np.zeros(intensity.shape)
  cond = (intensity < -500) | (softTissue == 1) & roi!=0
  dataCostSink[cond] = 1 * COST_AMPLIFIER
  # - TISSUE, 1
  dataCostSource = np.zeros(intensity.shape)
  cond = (intensity > 400) & (sheetness > 0) & roi!=0
  dataCostSource[cond] = 1 * COST_AMPLIFIER
  dataCostPixels = _pixelIdImage[roi!=0].flatten()
  flat_dataCostSink = dataCostSink[roi!=0].flatten()
  flat_dataCostSource = dataCostSource[roi!=0].flatten()
  # initializeNeighbours prep
  Xcenters, XFromCenter, XToCenter = SheetnessBasedSmoothCost(pixelLeft  = _pixelIdImage[:, :, :-1],
                                                              pixelRight = _pixelIdImage[:, :, 1:],
                                                              shtnLeft  = sheetness[:,:,:-1],
                                                              shtnRight = sheetness[:,:,1:],
                                                              COST_AMPLIFIER = COST_AMPLIFIER,
                                                              alpha = alpha)
  Ycenters, YFromCenter, YToCenter = SheetnessBasedSmoothCost(pixelLeft  = _pixelIdImage[:, :-1, :],
                                                              pixelRight = _pixelIdImage[:, 1:, :],
                                                              shtnLeft  = sheetness[:,:-1,:],
                                                              shtnRight = sheetness[:,1:,:],
                                                              COST_AMPLIFIER = COST_AMPLIFIER,
                                                              alpha = alpha)
  Zcenters, ZFromCenter, ZToCenter = SheetnessBasedSmoothCost(pixelLeft  = _pixelIdImage[:-1,:,:],
                                                              pixelRight = _pixelIdImage[1:,:,:],
                                                              shtnLeft  = sheetness[:-1,:,:],
                                                              shtnRight = sheetness[1:,:,:],
                                                              COST_AMPLIFIER = COST_AMPLIFIER,
                                                              alpha = alpha)
  CentersPixels = np.concatenate([Zcenters[0], Ycenters[0], Xcenters[0] ])
  NeighborsPixels = np.concatenate([Zcenters[1], Ycenters[1], Xcenters[1] ])
  _totalNeighbors = len(NeighborsPixels)
  flat_smoothCostFromCenter = np.concatenate([ZFromCenter, YFromCenter, XFromCenter ])
  flat_smoothCostToCenter = np.concatenate([ZToCenter, YToCenter, XToCenter ])
  # Call Maxflow
  uint_gcresult = GraphCutSupport.RunGraphCut(_totalPixelsInROI,
                                              np.ascontiguousarray(dataCostPixels, dtype=np.uint32),
                                              np.ascontiguousarray(flat_dataCostSource, dtype=np.uint32),
                                              np.ascontiguousarray(flat_dataCostSink, dtype=np.uint32),
                                              _totalNeighbors,
                                              np.ascontiguousarray(CentersPixels, dtype=np.uint32),
                                              np.ascontiguousarray(NeighborsPixels, dtype=np.uint32),
                                              np.ascontiguousarray(flat_smoothCostFromCenter, dtype=np.uint32),
                                              np.ascontiguousarray(flat_smoothCostToCenter, dtype=np.uint32)
                                              )
  _labelIdImage = _pixelIdImage
  _labelIdImage[roi!=0] = uint_gcresult
  _labelIdImage[roi==0] = 0
  _labelIdImage = np.asarray(_labelIdImage, dtype=np.uint8)
  gcresult = itk.GetImageFromArray(_labelIdImage)
  # _labelIdImage.tolist() == (itk.GetArrayFromImage(cpp_gc)).tolist()
  gcresult.SetOrigin(imgObj.GetOrigin())
  gcresult.SetSpacing(imgObj.GetSpacing())
  gcresult.SetDirection(imgObj.GetDirection())
  return gcresult

#%%
############################
# BONE SEPARATION FUNCTION #
############################
def opening(labelImage,
            radius,
            ImageType,
            d=3):
  StructuringElementType = itk.FlatStructuringElement[d]
  structuringElement = StructuringElementType.Ball(radius)
  OpeningFilterType = itk.BinaryMorphologicalOpeningImageFilter[itk.Image[ImageType,d], itk.Image[ImageType,d], StructuringElementType]
  OpeningFilter = OpeningFilterType.New()
  OpeningFilter.SetKernel(structuringElement)
  OpeningFilter.SetInput( labelImage )
  OpeningFilter.Update()
  return OpeningFilter.GetOutput()


def erosion(labelImage,
            radius,
            ImageType,
            d=3,
            valueToErode = 1):
  StructuringElementType = itk.FlatStructuringElement[d]
  structuringElement = StructuringElementType.Ball(radius)
  ErodeFilterType = itk.BinaryErodeImageFilter[itk.Image[ImageType,d], itk.Image[ImageType,d], StructuringElementType]
  erosionFilter = ErodeFilterType.New()
  erosionFilter.SetKernel(structuringElement)
  erosionFilter.SetErodeValue(valueToErode)
  erosionFilter.SetInput( labelImage )
  erosionFilter.Update()
  return erosionFilter.GetOutput()

def SmoothnessCostFunction(pixelLeft,
                           pixelRight):
  cond = (pixelLeft > -1) & (pixelRight > -1)
  smoothCostFromCenter = np.ones(pixelLeft[cond].shape)
  smoothCostToCenter   = np.ones(pixelLeft[cond].shape)
  return (pixelLeft[cond], pixelRight[cond]), smoothCostFromCenter, smoothCostToCenter

def RefineSegmentation(islandImage,
                       subIslandLabels,
                       ROI):
  # assignIdsToPixels
  IslandsValues = itk.GetArrayFromImage(islandImage)
  _pixelIdImage = np.zeros(IslandsValues.shape)
  roi = itk.GetArrayFromImage(ROI)
  _pixelIdImage[roi==0] = -1
  _totalPixelsInROI = np.sum(roi!=0)
  _pixelIdImage[roi!=0] = range(_totalPixelsInROI)
  # SheetnessBasedDataCost_compute for initializeDataCosts prep
  # 0
  dataCostSink = np.zeros(IslandsValues.shape)
  cond = (IslandsValues == subIslandLabels[1]) & roi!=0
  dataCostSink[cond] = 1000
  # 1
  dataCostSource = np.zeros(IslandsValues.shape)
  cond = (IslandsValues == subIslandLabels[0]) & roi!=0
  dataCostSource[cond] = 1000
  dataCostPixels = _pixelIdImage[roi!=0].flatten()
  flat_dataCostSink = dataCostSink[roi!=0].flatten()
  flat_dataCostSource = dataCostSource[roi!=0].flatten()
  # initializeNeighbours prep
  Xcenters, XFromCenter, XToCenter = SmoothnessCostFunction(pixelLeft  = _pixelIdImage[:, :, :-1],
                                                            pixelRight = _pixelIdImage[:, :, 1:])
  Ycenters, YFromCenter, YToCenter = SmoothnessCostFunction(pixelLeft  = _pixelIdImage[:, :-1, :],
                                                            pixelRight = _pixelIdImage[:, 1:, :])
  Zcenters, ZFromCenter, ZToCenter = SmoothnessCostFunction(pixelLeft  = _pixelIdImage[:-1,:,:],
                                                            pixelRight = _pixelIdImage[1:,:,:])
  CentersPixels = np.concatenate([Zcenters[0], Ycenters[0], Xcenters[0] ])
  NeighborsPixels = np.concatenate([Zcenters[1], Ycenters[1], Xcenters[1] ])
  _totalNeighbors = len(NeighborsPixels)
  flat_smoothCostFromCenter = np.concatenate([ZFromCenter, YFromCenter, XFromCenter ])
  flat_smoothCostToCenter = np.concatenate([ZToCenter, YToCenter, XToCenter ])
  # Call Maxflow
  uint_gcresult = GraphCutSupport.RunGraphCut(_totalPixelsInROI,
                                              np.ascontiguousarray(dataCostPixels, dtype=np.uint32),
                                              np.ascontiguousarray(flat_dataCostSource, dtype=np.uint32),
                                              np.ascontiguousarray(flat_dataCostSink, dtype=np.uint32),
                                              _totalNeighbors,
                                              np.ascontiguousarray(CentersPixels, dtype=np.uint32),
                                              np.ascontiguousarray(NeighborsPixels, dtype=np.uint32),
                                              np.ascontiguousarray(flat_smoothCostFromCenter, dtype=np.uint32),
                                              np.ascontiguousarray(flat_smoothCostToCenter, dtype=np.uint32)
                                              )
  _labelIdImage = _pixelIdImage
  _labelIdImage[roi!=0] = uint_gcresult
  _labelIdImage[roi==0] = 0
  _labelIdImage = np.asarray(_labelIdImage, dtype=np.uint8)
  gcresult = itk.GetImageFromArray(_labelIdImage)
  gcresult.SetOrigin(islandImage.GetOrigin())
  gcresult.SetSpacing(islandImage.GetSpacing())
  gcresult.SetDirection(islandImage.GetDirection())
  return gcresult


def isIslandWithinDistance(image,
                           distanceImage,
                           label,
                           maxDistance
                          ):
  values = itk.GetArrayFromImage(image)
  dist_values = itk.GetArrayFromImage(distanceImage)
  return np.any(dist_values[values == label] < maxDistance)


def distanceMapByFastMarcher(image,
                             objectLabel,
                             stoppingValue,
                             ImageType
                            ):
  FastMarchingImageFilter = itk.FastMarchingImageFilter[ImageType, ImageType]
  fastMarcher = FastMarchingImageFilter.New()
  fastMarcher.SetOutputSize(image.GetLargestPossibleRegion().GetSize())
  fastMarcher.SetOutputOrigin(image.GetOrigin() )
  fastMarcher.SetOutputSpacing(image.GetSpacing() )
  fastMarcher.SetOutputDirection(image.GetDirection() )
  fastMarcher.SetSpeedConstant(1.0)
  if (stoppingValue > 0):
    fastMarcher.SetStoppingValue(stoppingValue)
  NodeType = itk.LevelSetNode.F3
  FastMarchingNodeContainer = itk.VectorContainer[itk.UI, NodeType]
  TrialIndexes = np.array(np.where(itk.GetArrayFromImage(image) == objectLabel)).T
  seeds = FastMarchingNodeContainer.New()
  seeds.Initialize()
  for Idx in TrialIndexes:
    node = seeds.CreateElementAt(seeds.Size())
    node.SetValue(0.)
    node.SetIndex(Idx[::-1].tolist())
  fastMarcher.SetTrialPoints(seeds)
  fastMarcher.Update()
  return fastMarcher.GetOutput()

def duplicate(img,
              ImageType):
  DuplicatorType = itk.ImageDuplicator[ImageType]
  duplicator = DuplicatorType.New()
  duplicator.SetInputImage( img )
  duplicator.Update()
  return duplicator.GetOutput()

#%%
# Uncomment when needed:
import matplotlib.pylab as plb
def showSome(imgObj, idx = 0):
  prova = itk.GetArrayFromImage(imgObj)
  plb.imshow(prova[idx,:,:])
#
# def Read3DNifti(fn, t = 'unsigned char'):
#   nifti_obj = itk.NiftiImageIO.New()
#   set_type = itk.ctype(t)
#   reader_type = itk.Image[set_type,3]
#   reader = itk.ImageFileReader[reader_type].New()
#   reader.SetFileName(fn)
#   reader.SetImageIO(nifti_obj)
#   reader.Update()
#   return reader.GetOutput()
#%%
##########
# Parser #
##########

def parse_args ():
  description = "GraphCut-based Femur Unsupervised 3D Segmentation"
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--indir',
                      dest='indir',
                      required=True,
                      type=str,
                      action='store',
                      help='Input DICOMs directory' )
  parser.add_argument('--outdir',
                      dest='outdir',
                      required=False,
                      type=str,
                      action='store',
                      help='Output DICOMs directory',
                      default='')
  args = parser.parse_args()
  return args


#%%
global inputCT
global Inputmetadata

def GCAutoSegm (sigmaSmallScale,
                sigmasLargeScale_number,
                sigmasLargeScale_min,
                sigmasLargeScale_step,
                lowerThreshold,
                upperThreshold,
                binThres_criteria,
                bone_criteriaA,
                bone_criteriaB,
                bone_smScale_criteria,
                autoROI_criteriaLow,
                autoROI_criteriaHigh,
                m_DiffusionFilter_sigma,
                m_SubstractFilter_scale,
                EROSION_RADIUS,
                MAX_DISTANCE_FOR_ADJACENT_BONES):

  global inputCT
  global Inputmetadata

  # ShowIdx = 400
  sigmasLargeScale_max = sigmasLargeScale_min + sigmasLargeScale_number*sigmasLargeScale_step
  sigmasLargeScale = np.arange(sigmasLargeScale_min, sigmasLargeScale_max, sigmasLargeScale_step)

  # Useful shortcut
  ShortType = itk.ctype('short')
  UShortType = itk.ctype('unsigned short')
  UCType = itk.ctype('unsigned char')
  FType = itk.ctype('float')
  ULType = itk.ctype('unsigned long')
  FloatImageType = itk.Image[FType,3]
  ShortImageType = itk.Image[ShortType,3]
  UCImageType = itk.Image[UCType,3]
  ULImageType = itk.Image[ULType,3]

  print("Preprocessing")
  ##################
  # PRE-PROCESSING #
  ##################
  print("Thresholding input image")
  thresholdedInputCT = thresholding(duplicate(inputCT, ShortImageType), lowerThreshold, upperThreshold) # Checked
  # showSome(thresholdedInputCT, ShowIdx)

  smallScaleSheetnessImage = multiscaleSheetness(multiScaleInput = castImage(thresholdedInputCT, OutputType=FloatImageType),
                                                 scales = [sigmaSmallScale],
                                                 SmoothingImageType = FloatImageType)
  # showSome(smallScaleSheetnessImage,ShowIdx)
  print("Estimating soft-tissue voxels")
  smScale_bin = binaryThresholding(inputImage = smallScaleSheetnessImage,
                                   lowerThreshold = -binThres_criteria,
                                   upperThreshold = binThres_criteria,
                                   outputImageType = UCType)
  # showSome(smScale_bin,ShowIdx)
  smScale_cc = ConnectedComponents(inputImage = smScale_bin,
                                   outputImageType = ULType
                                  )
  # showSome(smScale_cc,ShowIdx)
  smScale_rc = RelabelComponents(inputImage = smScale_cc,
                                 outputImageType=None)
  # showSome(smScale_rc,ShowIdx)
  softTissueEstimation = binaryThresholding(inputImage = smScale_rc,
                                            lowerThreshold = 1, # Extract largest non-zero connected component
                                            upperThreshold = 1)
  # showSome(softTissueEstimation, ShowIdx)
  print("Estimating bone voxels")
  boneEstimation = itk.GetArrayFromImage(inputCT)
  smScale = itk.GetArrayFromImage(smallScaleSheetnessImage)

  boneCondition = (boneEstimation > bone_criteriaA) | (boneEstimation > bone_criteriaB) & (smScale > bone_smScale_criteria)
  boneEstimation[boneCondition] = 1
  boneEstimation[np.logical_not(boneCondition)] = 0
  print("Computing ROI from bone estimation using Chamfer Distance")
  # Old version:
  # boneDist = distance_transform_cdt(boneEstimation.astype(np.int64),
  #                                   metric='taxicab',
  #                                   return_distances=True).astype(np.float64)
  boneDist = DistanceTransform(boneEstimation.astype(np.int32)).astype(np.float32)
  boneDist = itk.GetImageFromArray(boneDist)
  boneDist.SetOrigin(inputCT.GetOrigin())
  boneDist.SetSpacing(inputCT.GetSpacing())
  boneDist.SetDirection(inputCT.GetDirection())
  # showSome(boneDist, ShowIdx)
  autoROI  = binaryThresholding(inputImage = boneDist,
                                lowerThreshold = autoROI_criteriaLow,
                                upperThreshold = autoROI_criteriaHigh,
                                outputImageType = UCType)
  # showSome(autoROI, ShowIdx)
  print("Unsharp masking")
  InputCT_float = castImage(inputCT, OutputType=FloatImageType)
  # I*G (discrete gauss)
  m_DiffusionFilter = Gaussian(GaussInput = InputCT_float,
                               sigma = m_DiffusionFilter_sigma)
  # showSome(m_DiffusionFilter, ShowIdx)
  # I - (I*G)
  m_SubstractFilter = substract(InputCT_float, m_DiffusionFilter)
  # showSome(m_SubstractFilter, ShowIdx)
  # k(I-(I*G))
  m_MultiplyFilter = linearTransform(m_SubstractFilter,
                                     scale = m_SubstractFilter_scale,
                                     shift = 0.)
  # showSome(m_MultiplyFilter, ShowIdx)
  # I+k*(I-(I*G))
  inputCTUnsharpMasked = add(InputCT_float, m_MultiplyFilter)
  # showSome(inputCTUnsharpMasked, ShowIdx)
  print("Computing multiscale sheetness measure at %d scales" % len(sigmasLargeScale))
  Sheetness = multiscaleSheetness(multiScaleInput=inputCTUnsharpMasked,
                                  scales = sigmasLargeScale,
                                  SmoothingImageType = FloatImageType,
                                  roi = autoROI)
  # showSome(Sheetness, ShowIdx)
  ###########
  # Segment #
  ###########
  print("Segmentation")
  gcResult = Segmentation(imgObj = inputCT,
                          softEst = softTissueEstimation,
                          sht = Sheetness,
                          ROI = autoROI
                          )
  # showSome(gcResult, ShowIdx)

  ###################
  # Bone-separation #
  ###################
  print("Bone Separation")

  print("Computing Connected Components")
  mainIslands = ConnectedComponents(inputImage = gcResult,
                                    outputImageType = ULType)
  # showSome(mainIslands, ShowIdx)
  print("Erosion + Connected Components, ball radius=%d"% EROSION_RADIUS)
  eroded_gc = erosion(gcResult, EROSION_RADIUS, UCType)
  # showSome(eroded_gc, ShowIdx)
  subIslands = ConnectedComponents(inputImage = eroded_gc,
                                   outputImageType = ULType)
  # showSome(subIslands, ShowIdx)
  print("Discovering main islands containg bottlenecks")
  mainArray = itk.GetArrayFromImage(mainIslands)
  subArray = itk.GetArrayFromImage(subIslands)
  main_labels, main_counts = np.unique(mainArray[mainArray!=0], return_counts=True)
  subIslandsInfo =  { l:np.unique(subArray[(subArray!=0) & (mainArray==l)], return_counts=True) for l in main_labels}
  activeStates = { l:np.array([ (cl/c > 0.001) and (cl > 100) for sl, cl in zip(subIslandsInfo[l][0],subIslandsInfo[l][1]) ]) for l, c in zip(main_labels, main_counts) }
  MainIslandsToProcess = [ l for l in main_labels if np.sum(activeStates[l])>=2 ]
  subIslandsSortedBySize = { l:subIslandsInfo[l][0][activeStates[l]][np.argsort(subIslandsInfo[l][1][activeStates[l]]) ] for l in MainIslandsToProcess }
  subIslandsPairs = []
  for l in MainIslandsToProcess:
    for idx in range(len(subIslandsSortedBySize[l])-1):
      subIsland = subIslandsSortedBySize[l][idx]
      print("Computing distance from sub-island %d"% subIsland)
      distance = distanceMapByFastMarcher(image = subIslands,
                                          objectLabel = subIsland,
                                          stoppingValue = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES + 1),
                                          ImageType = FloatImageType)
      for jdx in range(idx+1, len(subIslandsSortedBySize[l])):
        potentialAdjacentSubIsland = subIslandsSortedBySize[l][jdx]
        islandsAdjacent = isIslandWithinDistance(image = subIslands,
                                                 distanceImage = distance,
                                                 label = potentialAdjacentSubIsland,
                                                 maxDistance = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES)
                                                )
        subIslandsPairs += [ [l, subIsland, potentialAdjacentSubIsland] ] if islandsAdjacent else []
  print("Number of bottlenecks to be found: %d"% len(subIslandsPairs));
  for subI in subIslandsPairs:
    mainLabel, i1, i2 = map(int, subI)
    print("Identifying bottleneck between sub-islands %d and %d within main island %d"%(i1, i2, mainLabel))
    roiSub = binaryThresholding(inputImage = mainIslands,
                                lowerThreshold = mainLabel,
                                upperThreshold = mainLabel)
    gcOutput = RefineSegmentation(islandImage = subIslands,
                                  subIslandLabels=[i1, i2],
                                  ROI = roiSub)
    uniqueLabel = np.max(mainArray) + 1
    gcValues = itk.GetArrayFromImage(gcOutput)
    mainArray[gcValues==1] = uniqueLabel # result
  relabelled_mainIslands = itk.GetImageFromArray(mainArray)
  relabelled_mainIslands.SetOrigin(mainIslands.GetOrigin())
  relabelled_mainIslands.SetSpacing(mainIslands.GetSpacing())
  relabelled_mainIslands.SetDirection(mainIslands.GetDirection())
  finalResult = RelabelComponents(inputImage = relabelled_mainIslands,
                                  outputImageType = UCType)
  # showSome(finalResult, ShowIdx)
  return finalResult


#%%
def ShortSegm (sigmaSmallScale,
               sigmasLargeScale_number,
               sigmasLargeScale_min,
               sigmasLargeScale_step,
               lowerThreshold,
               upperThreshold,
               binThres_criteria,
               bone_criteriaA,
               bone_criteriaB,
               bone_smScale_criteria,
               autoROI_criteriaLow,
               autoROI_criteriaHigh,
               m_DiffusionFilter_sigma,
               m_SubstractFilter_scale):
  # ShowIdx = 400
  sigmasLargeScale_max = sigmasLargeScale_min + sigmasLargeScale_number*sigmasLargeScale_step
  sigmasLargeScale = np.arange(sigmasLargeScale_min, sigmasLargeScale_max, sigmasLargeScale_step)

  # Useful shortcut
  ShortType = itk.ctype('short')
  UShortType = itk.ctype('unsigned short')
  UCType = itk.ctype('unsigned char')
  FType = itk.ctype('float')
  ULType = itk.ctype('unsigned long')
  FloatImageType = itk.Image[FType,3]
  ShortImageType = itk.Image[ShortType,3]
  UCImageType = itk.Image[UCType,3]
  ULImageType = itk.Image[ULType,3]

  print("Preprocessing")
  ##################
  # PRE-PROCESSING #
  ##################
  print("Thresholding input image")
  thresholdedInputCT = thresholding(duplicate(inputCT, ShortImageType), lowerThreshold, upperThreshold) # Checked
  smallScaleSheetnessImage = multiscaleSheetness(multiScaleInput = castImage(thresholdedInputCT, OutputType=FloatImageType),
                                                 scales = [sigmaSmallScale],
                                                 SmoothingImageType = FloatImageType)
  print("Estimating soft-tissue voxels")
  smScale_bin = binaryThresholding(inputImage = smallScaleSheetnessImage,
                                   lowerThreshold = -binThres_criteria,
                                   upperThreshold = binThres_criteria,
                                   outputImageType = UCType)
  smScale_cc = ConnectedComponents(inputImage = smScale_bin,
                                   outputImageType = ULType
                                  )
  smScale_rc = RelabelComponents(inputImage = smScale_cc,
                                 outputImageType=None)
  softTissueEstimation = binaryThresholding(inputImage = smScale_rc,
                                            lowerThreshold = 1, # Extract largest non-zero connected component
                                            upperThreshold = 1)
  print("Estimating bone voxels")
  boneEstimation = itk.GetArrayFromImage(inputCT)
  smScale = itk.GetArrayFromImage(smallScaleSheetnessImage)

  boneCondition = (boneEstimation > bone_criteriaA) | (boneEstimation > bone_criteriaB) & (smScale > bone_smScale_criteria)
  boneEstimation[boneCondition] = 1
  boneEstimation[np.logical_not(boneCondition)] = 0
  print("Computing ROI from bone estimation using Chamfer Distance")
  boneDist = DistanceTransform(boneEstimation.astype(np.int32)).astype(np.float32)
  boneDist = itk.GetImageFromArray(boneDist)
  boneDist.SetOrigin(inputCT.GetOrigin())
  boneDist.SetSpacing(inputCT.GetSpacing())
  boneDist.SetDirection(inputCT.GetDirection())
  autoROI  = binaryThresholding(inputImage = boneDist,
                                lowerThreshold = autoROI_criteriaLow,
                                upperThreshold = autoROI_criteriaHigh,
                                outputImageType = UCType)
  print("Unsharp masking")
  InputCT_float = castImage(inputCT, OutputType=FloatImageType)
  m_DiffusionFilter = Gaussian(GaussInput = InputCT_float,
                               sigma = m_DiffusionFilter_sigma)
  m_SubstractFilter = substract(InputCT_float, m_DiffusionFilter)
  m_MultiplyFilter = linearTransform(m_SubstractFilter,
                                     scale = m_SubstractFilter_scale,
                                     shift = 0.)
  inputCTUnsharpMasked = add(InputCT_float, m_MultiplyFilter)
  print("Computing multiscale sheetness measure at %d scales" % len(sigmasLargeScale))
  Sheetness = multiscaleSheetness(multiScaleInput=inputCTUnsharpMasked,
                                  scales = sigmasLargeScale,
                                  SmoothingImageType = FloatImageType,
                                  roi = autoROI)
  ###########
  # Segment #
  ###########
  print("Segmentation")
  gcResult = Segmentation(imgObj = inputCT,
                          softEst = softTissueEstimation,
                          sht = Sheetness,
                          ROI = autoROI
                          )
  return gcResult

def softTissueMacro(sSS_temp,
                    BoundEst_temp,
                    EROSION_RADIUS_MARG,
                    smallScaleSheetnessImage,
                    Sheetness,
                    autoROI
                    ):
  smallScaleSheetnessImage_arr = itk.GetArrayFromImage(smallScaleSheetnessImage)
  sSS_temp[BoundEst_temp==1] = 0
  sSS = itk.GetImageFromArray(sSS_temp)
  eroded_sSS = erosion(sSS, EROSION_RADIUS_MARG, UCType)
  eroded_sSS_arr = np.logical_not(itk.GetArrayFromImage(eroded_sSS))
  marginal_regions = eroded_sSS_arr*smallScaleSheetnessImage_arr
  A_crit, B_crit = np.percentile(marginal_regions[marginal_regions!=0], [25,75])
  smScale_bin = binaryThresholding(inputImage = smallScaleSheetnessImage,
                                   lowerThreshold = A_crit,
                                   upperThreshold = B_crit,
                                   outputImageType = UCType)
  # showSome(smScale_bin,ShowIdx)
  smScale_cc = ConnectedComponents(inputImage = smScale_bin,
                                   outputImageType = ULType
                                  )
  # showSome(smScale_cc,ShowIdx)
  smScale_rc = RelabelComponents(inputImage = smScale_cc,
                                 outputImageType=None)
  # showSome(smScale_rc,ShowIdx)
  softTissueEstimation = binaryThresholding(inputImage = smScale_rc,
                                            lowerThreshold = 1, # Extract largest non-zero connected component
                                            upperThreshold = 1)
  gcResult_v2 = Segmentation(imgObj = inputCT,
                             softEst = softTissueEstimation,
                             sht = Sheetness,
                             ROI = autoROI
                             )
  return gcResult_v2, A_crit, B_crit


#%%


BoundariesEstimation = ShortSegm(sigmaSmallScale = 2.2040020874558834,
                                 sigmasLargeScale_number = 3,
                                 sigmasLargeScale_min = 1.8268748154005032,
                                 sigmasLargeScale_step = 0.13673590788886275,
                                 lowerThreshold = 568,
                                 upperThreshold = 703,
                                 binThres_criteria = 0.4537692833349022,
                                 bone_criteriaA = 523,
                                 bone_criteriaB = 122,
                                 bone_smScale_criteria = 0.32704595016587884,
                                 autoROI_criteriaLow = 0,
                                 autoROI_criteriaHigh = 14,
                                 m_DiffusionFilter_sigma = 46.0,
                                 m_SubstractFilter_scale = 0.9465600306044972)
MainBonesEstimation = ShortSegm(sigmaSmallScale = 2.3431706848114375,
                                sigmasLargeScale_number = 7,
                                sigmasLargeScale_min = 0.5611119614464578,
                                sigmasLargeScale_step = 0.2062579541371359,
                                lowerThreshold = 25,
                                upperThreshold = 600,
                                binThres_criteria = 0.05,
                                bone_criteriaA = 400,
                                bone_criteriaB = 250,
                                bone_smScale_criteria = 0.5080780340864003,
                                autoROI_criteriaLow  = 0,
                                autoROI_criteriaHigh = 30,
                                m_DiffusionFilter_sigma = 0.8316166698338118,
                                m_SubstractFilter_scale = 10.)
BoundEst = BoundariesEstimation
BonesEst = MainBonesEstimation

def Erosion2D (Img3D, radius=3):
  structure = np.ones((radius,radius))
  Img3D_arr = itk.GetArrayFromImage(Img3D)
  for i in range(Img3D_arr.shape[0]) :
    Img3D_arr[i,:,:] = binary_erosion(Img3D_arr[i,:,:], structure)
  Img3D_eroded = itk.GetImageFromArray(Img3D_arr)
  Img3D_eroded.SetOrigin(Img3D.GetOrigin())
  Img3D_eroded.SetSpacing(Img3D.GetSpacing())
  Img3D_eroded.SetDirection(Img3D.GetDirection())
  return Img3D_eroded

def Opening2D(Img3D, radius=3):
  structure = skimage.morphology.disk(radius)
  Img3D_arr = itk.GetArrayFromImage(Img3D)
  for i in range(Img3D_arr.shape[0]) :
    Img3D_arr[i,:,:] = skimage.morphology.binary_opening(Img3D_arr[i,:,:], structure)
  Img3D_opened = itk.GetImageFromArray(Img3D_arr)
  Img3D_opened.SetOrigin(Img3D.GetOrigin())
  Img3D_opened.SetSpacing(Img3D.GetSpacing())
  Img3D_opened.SetDirection(Img3D.GetDirection())
  return Img3D_opened

def Closing2D(Img3D, radius=3):
  structure = skimage.morphology.disk(radius)
  Img3D_arr = itk.GetArrayFromImage(Img3D)
  for i in range(Img3D_arr.shape[0]) :
    Img3D_arr[i,:,:] = skimage.morphology.binary_closing(Img3D_arr[i,:,:], structure)
  Img3D_opened = itk.GetImageFromArray(Img3D_arr)
  Img3D_opened.SetOrigin(Img3D.GetOrigin())
  Img3D_opened.SetSpacing(Img3D.GetSpacing())
  Img3D_opened.SetDirection(Img3D.GetDirection())
  return Img3D_opened

def WatershedSegm(Img, markers, ImageTypeInput=FloatImageType, ImageTypeOutput=ShortImageType, conn=False):
  WatImageFilter = itk.MorphologicalWatershedFromMarkersImageFilter[ImageTypeInput, ImageTypeOutput]
  WatImage = WatImageFilter.New()
  WatImage.SetInput1(Img)
  WatImage.SetInput2(markers)
  WatImage.SetFullyConnected(conn)
  WatImage.Update()
  res = WatImage.GetOutput()
  return res

from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
Img = castImage(BonesEst, ShortImageType)
# distance = SignedMaurerDistanceMap(castImage(Img, FloatImageType),
#                                    ImageType=FloatImageType,
#                                    spacing=True,
#                                    value=1)
# distance_arr = itk.GetArrayFromImage(distance)
# distance_arr[itk.GetArrayFromImage(Img)==0] = 0
# plb.imshow(distance_arr[254,:,:])
# local_maxi = peak_local_max(-distance_arr, indices=False, footprint=np.ones((3, 3, 3)), labels=BonesEst_arr)
# markers = ndi.label(local_maxi)[0]
plb.imshow(RoughBones[240,:,:])
# local_maxi = itk.GetArrayFromImage(Erosion2D(gcResult, 3))


def WatershedSplit(GCImg, RoughBones, radius):
  RoughBones = castImage(RoughBones, ShortImageType)
  RoughBones_arr = itk.GetArrayFromImage(RoughBones)
  RoughBones_arr_copy = RoughBones_arr.copy()
  GCImg = castImage(GCImg, OutputType=ShortImageType)
  gcResult_Bound_arr = GetBoundaries(GCImg, ShortImageType, 0, 1)
  RoughBones_arr[gcResult_Bound_arr==1] = 0
  MaskedRoughBones = itk.GetImageFromArray(RoughBones_arr)
  MaskedRoughBones.SetOrigin(inputCT.GetOrigin())
  MaskedRoughBones.SetSpacing(inputCT.GetSpacing())
  MaskedRoughBones.SetDirection(inputCT.GetDirection())
  MaskedRoughBones = castImage(MaskedRoughBones, UCImageType)
  eroded_RoughBones = erosion(MaskedRoughBones, radius, UCType)
  # local_maxi = itk.GetArrayFromImage(eroded_RoughBones)
  # plb.imshow(local_maxi[240,:,:])
  markers_itk = ConnectedComponents(inputImage = eroded_RoughBones, outputImageType = ULType)
  # showSome(markers_itk, 240)
  # np.unique(itk.GetArrayFromImage(markers_itk), return_counts=True)
  markers_itk = castImage(markers_itk, ShortImageType)
  markers_itk.SetOrigin(RoughBones.GetOrigin())
  markers_itk.SetSpacing(RoughBones.GetSpacing())
  markers_itk.SetDirection(RoughBones.GetDirection())
  res = WatershedSegm(castImage(RoughBones, FloatImageType), markers_itk, conn=True)
  res_arr = itk.GetArrayFromImage(res)
  res_arr[RoughBones_arr_copy==0] = 0
  # plb.imshow(res_arr[247,:,:])
  # showSome(inputCT, 247)
  SegRes = itk.GetImageFromArray(res_arr)
  SegRes.SetOrigin(inputCT.GetOrigin())
  SegRes.SetSpacing(inputCT.GetSpacing())
  SegRes.SetDirection(inputCT.GetDirection())
  return SegRes

# res_arr = itk.GetArrayFromImage(res)
# res_arr[RoughBones_arr_copy==0] = 0
# plb.imshow(res_arr[240,:,:])
# temp = np.unique(res_arr, return_counts=True)
# np.unique(res_arr[240,:,:])
# #%%
# prova_show = res_arr.copy()
# cond_res = (res_arr==1) | (res_arr==15)
# prova_show[ cond_res ] = 1
# prova_show[ np.logical_not(cond_res) ] = 0
# plb.imshow(prova_show[240,:,:])
# showSome(inputCT, 240)

#%%


def GCAutoSegm (BoundEst,
                BonesEst):
  # ShowIdx = 240
  sigmaSmallScale = 2.3431706848114375
  # sigmaSmallScale = 1.5
  sigmasLargeScale_number = 7
  sigmasLargeScale_min = 0.5611119614464578
  sigmasLargeScale_step = 0.2062579541371359
  sigmasLargeScale_max = sigmasLargeScale_min + sigmasLargeScale_number*sigmasLargeScale_step
  sigmasLargeScale = np.arange(sigmasLargeScale_min, sigmasLargeScale_max, sigmasLargeScale_step)
  EROSION_RADIUS_MARG = 3
  EROSION_RADIUS = 3
  MAX_DISTANCE_FOR_ADJACENT_BONES = 15
  autoROI_criteriaLow = 0
  autoROI_criteriaHigh = 30
  m_DiffusionFilter_sigma = 0.83
  m_SubstractFilter_scale = 10.

  # Useful shortcut
  ShortType = itk.ctype('short')
  UShortType = itk.ctype('unsigned short')
  UCType = itk.ctype('unsigned char')
  FType = itk.ctype('float')
  ULType = itk.ctype('unsigned long')
  FloatImageType = itk.Image[FType,3]
  ShortImageType = itk.Image[ShortType,3]
  UCImageType = itk.Image[UCType,3]
  ULImageType = itk.Image[ULType,3]

  print("Preprocessing")
  ##################
  # PRE-PROCESSING #
  ##################
  print("Thresholding input image")
  BoundEst_arr = itk.GetArrayFromImage(BoundEst)
  BonesEst_arr = itk.GetArrayFromImage(BonesEst)
  EstUnion_arr = BoundEst_arr | BonesEst_arr
  EstUnion_arr_cond = EstUnion_arr==1
  inputCT_arr  = itk.GetArrayFromImage(inputCT)
  thresholdedInputCT = EstUnion_arr*inputCT_arr
  thresholdedInputCT = itk.GetImageFromArray(thresholdedInputCT)
  thresholdedInputCT.SetOrigin(inputCT.GetOrigin())
  thresholdedInputCT.SetSpacing(inputCT.GetSpacing())
  thresholdedInputCT.SetDirection(inputCT.GetDirection())
  # thresholdedInputCT = thresholding(duplicate(inputCT, ShortImageType), int(lowerThreshold), int(upperThreshold)) # Checked
  # showSome(thresholdedInputCT, ShowIdx)
  smallScaleSheetnessImage = multiscaleSheetness(multiScaleInput = castImage(thresholdedInputCT, OutputType=FloatImageType),
                                                 scales = [sigmaSmallScale],
                                                 SmoothingImageType = FloatImageType)
  # showSome(smallScaleSheetnessImage,ShowIdx)
  smallScaleSheetnessImage_arr = itk.GetArrayFromImage(smallScaleSheetnessImage)
  sSS_temp = BonesEst_arr
  sSS_temp[BoundEst_arr==1] = 0
  sSS = itk.GetImageFromArray(sSS_temp)
  eroded_sSS = erosion(sSS, EROSION_RADIUS_MARG, UCType)
  eroded_sSS_arr = np.logical_not(itk.GetArrayFromImage(eroded_sSS))
  marginal_regions = eroded_sSS_arr*smallScaleSheetnessImage_arr
  A_crit, B_crit = np.percentile(marginal_regions[marginal_regions!=0], [25,75])

  print("Estimating soft-tissue voxels")
  smScale_bin = binaryThresholding(inputImage = smallScaleSheetnessImage,
                                   lowerThreshold = A_crit,
                                   upperThreshold = B_crit,
                                   outputImageType = UCType)
  # showSome(smScale_bin,ShowIdx)
  smScale_cc = ConnectedComponents(inputImage = smScale_bin,
                                   outputImageType = ULType
                                  )
  # showSome(smScale_cc,ShowIdx)
  smScale_rc = RelabelComponents(inputImage = smScale_cc,
                                 outputImageType=None)
  # showSome(smScale_rc,ShowIdx)
  softTissueEstimation = binaryThresholding(inputImage = smScale_rc,
                                            lowerThreshold = 1, # Extract largest non-zero connected component
                                            upperThreshold = 1)
  # showSome(softTissueEstimation, ShowIdx)
  print("Estimating bone voxels")
  boneEstimation = BonesEst_arr
  # smScale = itk.GetArrayFromImage(smallScaleSheetnessImage)

  # boneCondition = (boneEstimation > bone_criteriaA) | (boneEstimation > bone_criteriaB) & (smScale > bone_smScale_criteria)
  # boneEstimation[boneCondition] = 1
  # boneEstimation[np.logical_not(boneCondition)] = 0
  print("Computing ROI from bone estimation using Chamfer Distance")
  # Old version:
  # boneDist = distance_transform_cdt(boneEstimation.astype(np.int64),
  #                                   metric='taxicab',
  #                                   return_distances=True).astype(np.float64)
  boneDist = DistanceTransform(boneEstimation.astype(np.int32)).astype(np.float32)
  boneDist = itk.GetImageFromArray(boneDist)
  boneDist.SetOrigin(inputCT.GetOrigin())
  boneDist.SetSpacing(inputCT.GetSpacing())
  boneDist.SetDirection(inputCT.GetDirection())
  # showSome(boneDist, ShowIdx)
  autoROI  = binaryThresholding(inputImage = boneDist,
                                lowerThreshold = autoROI_criteriaLow,
                                upperThreshold = autoROI_criteriaHigh,
                                outputImageType = UCType)
  # showSome(autoROI, ShowIdx)
  print("Unsharp masking")
  InputCT_float = castImage(inputCT, OutputType=FloatImageType)
  # I*G (discrete gauss)
  m_DiffusionFilter = Gaussian(GaussInput = InputCT_float,
                               sigma = m_DiffusionFilter_sigma)
  # showSome(m_DiffusionFilter, ShowIdx)
  # I - (I*G)
  m_SubstractFilter = substract(InputCT_float, m_DiffusionFilter)
  # showSome(m_SubstractFilter, ShowIdx)
  # k(I-(I*G))
  m_MultiplyFilter = linearTransform(m_SubstractFilter,
                                     scale = m_SubstractFilter_scale,
                                     shift = 0.)
  # showSome(m_MultiplyFilter, ShowIdx)
  # I+k*(I-(I*G))
  inputCTUnsharpMasked = add(InputCT_float, m_MultiplyFilter)
  # showSome(inputCTUnsharpMasked, ShowIdx)
  print("Computing multiscale sheetness measure at %d scales" % len(sigmasLargeScale))
  Sheetness = multiscaleSheetness(multiScaleInput=inputCTUnsharpMasked,
                                  scales = sigmasLargeScale,
                                  SmoothingImageType = FloatImageType,
                                  roi = autoROI)
  # showSome(Sheetness, ShowIdx)
  ###########
  # Segment #
  ###########
  print("Segmentation")
  # gcResult = gcResult_backup
  gcResult = Segmentation(imgObj = inputCT,
                          softEst = softTissueEstimation,
                          sht = Sheetness,
                          ROI = autoROI,
                          COST_AMPLIFIER = 1e3,
                          alpha=5)

  # BoundEst_arr_v2 = itk.GetArrayFromImage(gcResult)
  # A_old = A_crit
  # B_old = B_crit
  # while True:
  #   res, A_new, B_new = softTissueMacro(sSS_temp = BonesEst_arr,
  #                                       BoundEst_temp = BoundEst_arr_v2,
  #                                       EROSION_RADIUS_MARG=EROSION_RADIUS_MARG,
  #                                       smallScaleSheetnessImage=smallScaleSheetnessImage,
  #                                       Sheetness=Sheetness,
  #                                       autoROI=autoROI)
  #   if (np.abs(A_new-A_old)<0.01) and (np.abs(B_new-B_old)<0.01):
  #     break
  #   else:
  #     A_old = A_new
  #     B_old = B_new
  #     BoundEst_arr_v2 = itk.GetArrayFromImage(res)
  # showSome(res, ShowIdx)
  # gcResult_backup = res
  gcResult_backup = gcResult
  showSome(gcResult_backup, 205)
  # showSome(inputCT, ShowIdx)

  # showSome(gcResult_backup, 259)

  # gcResult_Bound_arr = GetBoundaries(castImage(gcResult_backup, OutputType=ShortImageType), ShortImageType, 0, 1)
  # gcResult_Bound = itk.GetImageFromArray(gcResult_Bound_arr)
  # BonesEst_Bound_arr = GetBoundaries(castImage(BonesEst, OutputType=ShortImageType), ShortImageType, 0, 1)
  gcResult_backup_arr = itk.GetArrayFromImage(gcResult_backup)

  full_img = BonesEst_arr[205,:,:].copy()
  image = gcResult_backup_arr[205,:,:].copy()
  full_img[image==1] = 0
  plb.imshow(full_img)
  labels = measure.label(full_img)
  plb.imshow(labels)
  regions = measure.regionprops(labels)
  markers = np.array([r.centroid for r in regions]).astype(np.uint16)
  marker_image = np.zeros_like(full_img, dtype=np.int64)
  marker_image[markers[:, 0], markers[:, 1]]  = np.arange(len(markers)) + 1
  distance_map = ndi.distance_transform_edt(1 - full_img)
  plb.imshow(distance_map)
  filled = watershed(distance_map, markers=marker_image, mask=BonesEst_arr[205,:,:])
  filled_labs, filled_counts = np.unique(filled, return_counts=True)
  plb.imshow(filled)
  plb.imshow(image)
  image_tmp = filled.copy()
  image_tmp[image==1] = 0
  plb.imshow(image_tmp)
  # filled_connected = measure.label(filled != filled_labs[np.argmax(filled_counts)], background=0) + 1
  # plb.imshow(filled_connected)


  showSome(BonesEst, 198)

  ###################
  # Bone-separation #
  ###################
  print("Bone Separation")

  # 1st option
  # RoughBones = BonesEst_arr
  # gcResult_arr = itk.GetArrayFromImage(gcResult_backup)
  # RoughBones[gcResult_arr==1] = 0
  # gcResult = itk.GetImageFromArray(RoughBones)
  # gcResult.SetOrigin(inputCT.GetOrigin())
  # gcResult.SetSpacing(inputCT.GetSpacing())
  # gcResult.SetDirection(inputCT.GetDirection())
  # # gcResult = castImage(gcResult, ShortImageType)
  # showSome(gcResult,240)
  # showSome(gcResult_backup,250)
  # showSome(inputCT,250)
  # mainIslands = ConnectedComponents(inputImage = gcResult,
  #                                   outputImageType = ULType)

  print("Computing Connected Components")
  mainIslands = ConnectedComponents(inputImage = BonesEst,
                                    outputImageType = ULType)
  # showSome(mainIslands, ShowIdx)
  print("Erosion + Connected Components, ball radius=%d"% EROSION_RADIUS)
  # RoughBones[gcResult_Bound_arr==1] = 0
  # eroded_gc_pre = itk.GetImageFromArray(RoughBones)
  # # eroded_gc = erosion(gcResult, EROSION_RADIUS, UCType)
  # # eroded_gc = Erosion2D(gcResult, 9)
  # eroded_gc = eroded_gc_pre
  # showSome(eroded_gc,231)
  # subIslands = ConnectedComponents(inputImage = eroded_gc,
  # outputImageType = ULType)
  WaterRadius = 5
  watSeg = WatershedSplit(GCImg = gcResult_backup, RoughBones = BonesEst, radius = WaterRadius)
  subIslands = watSeg
  # showSome(subIslands, ShowIdx)
  mainArray = itk.GetArrayFromImage(mainIslands)
  subArray = itk.GetArrayFromImage(subIslands)


  print("Discovering main islands containg bottlenecks")
  # mainArray = itk.GetArrayFromImage(mainIslands)
  # subArray = itk.GetArrayFromImage(subIslands)
  # main_labels, main_counts = np.unique(mainArray[mainArray!=0], return_counts=True)
  # subIslandsInfo =  { l:np.unique(subArray[(subArray!=0) & (mainArray==l)], return_counts=True) for l in main_labels}
  # activeStates = { l:np.array([ (cl/c > 0.001) and (cl > 100) for sl, cl in zip(subIslandsInfo[l][0],subIslandsInfo[l][1]) ]) for l, c in zip(main_labels, main_counts) }
  # MainIslandsToProcess = [ l for l in main_labels if np.sum(activeStates[l])>=2 ]
  # subIslandsSortedBySize = { l:subIslandsInfo[l][0][activeStates[l]][np.argsort(subIslandsInfo[l][1][activeStates[l]]) ] for l in MainIslandsToProcess }
  # subIslandsPairs = []
  # for l in MainIslandsToProcess:
  #   for idx in range(len(subIslandsSortedBySize[l])-1):
  #     subIsland = subIslandsSortedBySize[l][idx]
  #     print("Computing distance from sub-island %d"% subIsland)
  #     distance = distanceMapByFastMarcher(image = subIslands,
  #                                         objectLabel = subIsland,
  #                                         stoppingValue = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES + 1),
  #                                         ImageType = FloatImageType)
  #     # showSome(distance, 240)
  #     for jdx in range(idx+1, len(subIslandsSortedBySize[l])):
  #       potentialAdjacentSubIsland = subIslandsSortedBySize[l][jdx]
  #       islandsAdjacent = isIslandWithinDistance(image = subIslands,
  #                                                distanceImage = distance,
  #                                                label = potentialAdjacentSubIsland,
  #                                                maxDistance = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES)
  #                                               )
  #       subIslandsPairs += [ [l, subIsland, potentialAdjacentSubIsland] ] if islandsAdjacent else []
  # print("Number of bottlenecks to be found: %d"% len(subIslandsPairs));
  # for subI in subIslandsPairs:
  #   mainLabel, i1, i2 = map(int, subI)
  #   print("Identifying bottleneck between sub-islands %d and %d within main island %d"%(i1, i2, mainLabel))
  #   roiSub = binaryThresholding(inputImage = mainIslands,
  #                               lowerThreshold = mainLabel,
  #                               upperThreshold = mainLabel)
  #   gcOutput = RefineSegmentation(islandImage = subIslands,
  #                                 subIslandLabels=[i1, i2],
  #                                 ROI = roiSub)
  #   uniqueLabel = np.max(mainArray) + 1
  #   gcValues = itk.GetArrayFromImage(gcOutput)
  #   mainArray[gcValues==1] = uniqueLabel # result
  # relabelled_mainIslands = itk.GetImageFromArray(mainArray)
  # relabelled_mainIslands.SetOrigin(mainIslands.GetOrigin())
  # relabelled_mainIslands.SetSpacing(mainIslands.GetSpacing())
  # relabelled_mainIslands.SetDirection(mainIslands.GetDirection())
  # finalResult = RelabelComponents(inputImage = relabelled_mainIslands,
  #                                 outputImageType = UCType)
  gcResult_backup_arr = itk.GetArrayFromImage(gcResult_backup)
  bones_labels = np.unique(subArray[gcResult_backup_arr==1])
  subLabs, subCounts = np.unique(subArray, return_counts=True)
  cond_first = bones_labels[bones_labels!=0]
  first_bone_idx = subLabs[cond_first][np.argmax(subCounts[cond_first])]
  cond_second = cond_first[cond_first!=first_bone_idx]
  second_bone_idx = subLabs[cond_second][np.argmax(subCounts[cond_second])]
  subArray_copy = subArray.copy()
  subArray_copy[(subArray_copy!=first_bone_idx) & (subArray_copy!=second_bone_idx)] = 0
  subArray_preOpen = itk.GetImageFromArray(subArray_copy)
  subArray_postOpen = opening(subArray_preOpen, 3, ShortType)
  subArray_copy = subArray.copy()
  cond_res = (subArray_copy==first_bone_idx) | (subArray_copy==second_bone_idx)
  Thinbones = itk.GetArrayFromImage(subArray_postOpen)
  subArray_copy[cond_res] = Thinbones[cond_res]
  subIslands = itk.GetImageFromArray(subArray_copy)

  print("Computing distance from bone 1")
  distance_1 = distanceMapByFastMarcher(image = subIslands,
                                        objectLabel = first_bone_idx,
                                        stoppingValue = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES + 1),
                                        ImageType = FloatImageType)
  print("Computing distance from bone 2")
  distance_2 = distanceMapByFastMarcher(image = subIslands,
                                        objectLabel = second_bone_idx,
                                        stoppingValue = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES + 1),
                                        ImageType = FloatImageType)
  dist_1 = itk.GetArrayFromImage(distance_1)
  dist_2 = itk.GetArrayFromImage(distance_2)

  from skimage import measure
  subArray_copy = subArray.copy()
  for idx in range(subArray_copy.shape[0]):
    tmp_img = subArray_copy[idx,:,:].copy()
    unique_labels = np.unique(tmp_img)
    unique_labels = unique_labels[(unique_labels!=0) & (unique_labels!=first_bone_idx) & (unique_labels!=second_bone_idx)]
    gc_guideline = gcResult_backup_arr[idx,:,:]
    tmp_sup = tmp_img.copy()
    tmp_sup[gc_guideline==1] = 0
    # plb.imshow(tmp_sup)
    dist_1_temp = dist_1[idx,:,:]
    dist_1_temp[gc_guideline==1] = 0
    dist_2_temp = dist_2[idx,:,:]
    dist_2_temp[gc_guideline==1] = 0
    conds_1 = np.array([np.any(dist_1_temp[tmp_sup == l] < 3) for l in unique_labels ])
    conds_2 = np.array([np.any(dist_2_temp[tmp_sup == l] < 3) for l in unique_labels ])
    for l in unique_labels[np.logical_not(conds_1 | conds_2)]:
      tmp_sup[tmp_sup==l] = 0
    subArray_copy[idx,:,:] = tmp_sup
    # plb.imshow(tmp_sup)
  plb.imshow(subArray_copy[259,:,:])

  # subArray_copy = subArray.copy()
  # subArray_copy[(subArray_copy!=first_bone_idx) & (subArray_copy!=second_bone_idx)] = 0
  # plb.imshow(subArray_copy[243,:,:])
  # plb.imshow(subArray_copy[245,:,:])
  showSome(gcResult_backup, 245)
  plb.imshow(subArray[250,:,:])

  main_labels, main_counts = np.unique(mainArray[mainArray!=0], return_counts=True)
  subIslandsInfo =  { l:np.unique(subArray[(subArray!=0) & (mainArray==l)], return_counts=True) for l in main_labels}
  # activeStates = { l:np.array([ (cl/c > 0.001) and (cl > 100) for sl, cl in zip(subIslandsInfo[l][0],subIslandsInfo[l][1]) ]) for l, c in zip(main_labels, main_counts) }
  # MainIslandsToProcess = [ l for l in main_labels if np.sum(activeStates[l])>=2 ]
  # subIslandsSortedBySize = { l:subIslandsInfo[l][0][activeStates[l]][np.argsort(subIslandsInfo[l][1][activeStates[l]]) ] for l in MainIslandsToProcess }
  # subIslandsPairs = []
  # subIslandsSplit = []
  # for l in MainIslandsToProcess:
  #   for idx in range(len(subIslandsSortedBySize[l])-1):
  #     subIsland = subIslandsSortedBySize[l][idx]
  #     print("Computing distance from sub-island %d"% subIsland)
  #     distance = distanceMapByFastMarcher(image = subIslands,
  #                                         objectLabel = subIsland,
  #                                         stoppingValue = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES + 1),
  #                                         ImageType = FloatImageType)
  #     # showSome(distance, 240)
  #     for jdx in range(idx+1, len(subIslandsSortedBySize[l])):
  #       potentialAdjacentSubIsland = subIslandsSortedBySize[l][jdx]
  #       if not ((potentialAdjacentSubIsland in [first_bone_idx, second_bone_idx]) or (subIsland in [first_bone_idx, second_bone_idx])):
  #         break
  #       islandsAdjacent = isIslandWithinDistance(image = subIslands,
  #                                                distanceImage = distance,
  #                                                label = potentialAdjacentSubIsland,
  #                                                maxDistance = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES)
  #                                               )
  #       subIslandsPairs += [ [l, subIsland, potentialAdjacentSubIsland] ] if islandsAdjacent else []
  #       subIslandsSplit += [ [l, subIsland, potentialAdjacentSubIsland] ] if not islandsAdjacent else []
  subIslandsPairs = []
  subIslandsSplit = []
  otherIslands = subLabs[(subLabs!=0) & (subLabs!=first_bone_idx) & (subLabs!=second_bone_idx) ]
  for subIsland in [first_bone_idx, second_bone_idx]:
    print("Computing distance from sub-island %d"% subIsland)
    distance = distanceMapByFastMarcher(image = subIslands,
                                        objectLabel = subIsland,
                                        stoppingValue = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES + 1),
                                        ImageType = FloatImageType)
    # showSome(distance, 240)
    for potentialAdjacentSubIsland in otherIslands:
      islandsAdjacent = isIslandWithinDistance(image = subIslands,
                                               distanceImage = distance,
                                               label = potentialAdjacentSubIsland,
                                               maxDistance = np.int(MAX_DISTANCE_FOR_ADJACENT_BONES)
                                              )
      subIslandsPairs += [ [l, subIsland, potentialAdjacentSubIsland] ] if islandsAdjacent else []
      subIslandsSplit += [ [l, subIsland, potentialAdjacentSubIsland] ] if not islandsAdjacent else []

  print("Number of bottlenecks to be found: %d"% len(subIslandsPairs))
  print("Number of not adjacent sub-islands: %d"% len(subIslandsSplit))

  countAdj = np.array(np.matrix(subIslandsPairs)[:,1].T)[0]
  countAdj = np.append(countAdj, np.array(np.matrix(subIslandsPairs)[:,2].T)[0])
  to_keep = np.unique(countAdj)
  subArray_copy = subArray.copy()
  for l in subLabs:
    if l in to_keep:
      continue
    else:
      print("Removing subisland %d"% l)
      subArray_copy[subArray_copy==l] = 0
  # to_remove = np.unique(countAdj[(countAdj!=first_bone_idx) & (countAdj!=second_bone_idx)])
  plb.imshow((subArray_copy*gcResult_backup_arr)[251,:,:])
  showSome(gcResult_backup, 2)
  plb.imshow(subArray[253,:,:])
  np.unique(subArray_copy)

  # countAdj = np.array(np.matrix(subIslandsPairs)[:,1].T)[0]
  # countAdj = np.append(countAdj, np.array(np.matrix(subIslandsPairs)[:,2].T)[0])
  # np.unique(countAdj, return_counts=True)
  # countAdj = np.array(np.matrix(subIslandsSplit)[:,1].T)[0]
  # countAdj = np.append(countAdj, np.array(np.matrix(subIslandsSplit)[:,2].T)[0])
  # np.unique(countAdj, return_counts=True)

  for subI in subIslandsPairs:
    mainLabel, i1, i2 = map(int, subI)
    print("Identifying bottleneck between sub-islands %d and %d within main island %d"%(i1, i2, mainLabel))
    roiSub = binaryThresholding(inputImage = mainIslands,
                                lowerThreshold = mainLabel,
                                upperThreshold = mainLabel)
    gcOutput = RefineSegmentation(islandImage = subIslands,
                                  subIslandLabels=[i1, i2],
                                  ROI = roiSub)
    uniqueLabel = np.max(mainArray) + 1
    gcValues = itk.GetArrayFromImage(gcOutput)
    mainArray[gcValues==1] = uniqueLabel # result
  relabelled_mainIslands = itk.GetImageFromArray(mainArray)
  relabelled_mainIslands.SetOrigin(mainIslands.GetOrigin())
  relabelled_mainIslands.SetSpacing(mainIslands.GetSpacing())
  relabelled_mainIslands.SetDirection(mainIslands.GetDirection())
  finalResult = RelabelComponents(inputImage = relabelled_mainIslands,
                                  outputImageType = UCType)

  # showSome(finalResult, ShowIdx)
  return finalResult
finalFemurs = GetFemurs(finalResult)
showSome(subIslands, 230)
showSome(inputCT, 240)
showSome(finalResult, 240)
np.unique(itk.GetArrayFromImage(finalResult)[231,:,:])
def MaskImg(ImgITK, cond):
  Img = itk.GetArrayFromImage(ImgITK)
  Img[np.logical_not(cond)] = 0
  ImgNew = itk.GetImageFromArray(Img)
  ImgNew.SetOrigin(ImgITK.GetOrigin())
  ImgNew.SetSpacing(ImgITK.GetSpacing())
  ImgNew.SetDirection(ImgITK.GetDirection())
  return ImgNew


#%%

def GetBoundaries(img, ImageType, back_value=0, fore_value=1):
  binaryContourImageFilterType = itk.BinaryContourImageFilter[ImageType,ImageType]
  binaryContourFilter = binaryContourImageFilterType.New()
  binaryContourFilter.SetInput(img)
  binaryContourFilter.SetBackgroundValue(back_value)
  binaryContourFilter.SetForegroundValue(fore_value)
  binaryContourFilter.Update()
  return itk.GetArrayFromImage(binaryContourFilter.GetOutput())

def SignedMaurerDistanceMap(img,
                            ImageType,
                            spacing = True,
                            value = 0,
                            inside = False,
                            squared = False):
  SignedMaurerDistanceMapImageFilterType = itk.SignedMaurerDistanceMapImageFilter[ImageType, ImageType]
  SignedMaurerDistanceMapImageFilter = SignedMaurerDistanceMapImageFilterType.New()
  SignedMaurerDistanceMapImageFilter.SetUseImageSpacing(spacing)
  SignedMaurerDistanceMapImageFilter.SetInput(img)
  SignedMaurerDistanceMapImageFilter.SetBackgroundValue(value)
  SignedMaurerDistanceMapImageFilter.SetInsideIsPositive(inside)
  SignedMaurerDistanceMapImageFilter.SetSquaredDistance(squared)
  SignedMaurerDistanceMapImageFilter.Update()
  return SignedMaurerDistanceMapImageFilter.GetOutput()

def LDMap(Input1, Input2, ImageType, spacing = True):
  distance_1 = itk.GetArrayFromImage(SignedMaurerDistanceMap(Input1, ImageType, spacing))
  distance_2 = itk.GetArrayFromImage(SignedMaurerDistanceMap(Input2, ImageType, spacing))
  A1 = (distance_1>1e-5).astype(np.float32)
  B1 = (distance_2>1e-5).astype(np.float32)
  LDMap_out = np.abs(A1 - B1) * np.maximum( distance_1, distance_2 )
  return LDMap_out



#%%

global ManualSegm
global bin_presence


all_arguments = { "GCSegm": ['sigmaSmallScale', 'sigmasLargeScale_number',
                             'sigmasLargeScale_min', 'sigmasLargeScale_step',
                             'lowerThreshold', 'upperThreshold',
                             'binThres_criteria', 'bone_criteriaA',
                             'bone_criteriaB', 'bone_smScale_criteria',
                             'autoROI_criteriaLow', 'autoROI_criteriaHigh',
                             'm_DiffusionFilter_sigma', 'm_SubstractFilter_scale',
                             'EROSION_RADIUS', 'MAX_DISTANCE_FOR_ADJACENT_BONES']}
# sigmaSmallScale = 1.5
# sigmasLargeScale_number = 1
# sigmasLargeScale_min = 0.6
# sigmasLargeScale_step = 0.2
# lowerThreshold = 25
# upperThreshold = 600
# binThres_criteria = 0.05
# bone_criteriaA = 400
# bone_criteriaB = 250
# bone_smScale_criteria = 0.6
# autoROI_criteriaLow  = 0
# autoROI_criteriaHigh = 30
# m_DiffusionFilter_sigma = 1.0
# m_SubstractFilter_scale = 10.
# EROSION_RADIUS = 3
# MAX_DISTANCE_FOR_ADJACENT_BONES = 15

# spaces = {
#            "GCSegm": [ Real(1., 3., "uniform", name='sigmaSmallScale'),
#                        Integer(2, 10, "identity", name='sigmasLargeScale_number'),
#                        Real(0.1, 2.0, "uniform", name='sigmasLargeScale_min'),
#                        Real(0.1, 1., "uniform", name='sigmasLargeScale_step'),
#                        Integer(0, 600, "identity", name='lowerThreshold'),
#                        Integer(600, 1000, "identity", name='upperThreshold'),
#                        Real(0.01, 1., "uniform", name='binThres_criteria'),
#                        Integer(300, 800, "identity", name='bone_criteriaA'),
#                        Integer(100, 299, "identity", name='bone_criteriaB'),
#                        Real(0.1, 1., "uniform", name='bone_smScale_criteria'),
#                        Integer(0, 29, "identity", name='autoROI_criteriaLow'),
#                        Integer(30, 60, "identity", name='autoROI_criteriaHigh'),
#                        Real(0.1, 2.0, "uniform", name='m_DiffusionFilter_sigma'),
#                        Real(1., 30., "uniform", name='m_SubstractFilter_scale'),
#                        Categorical([3,5,7,9], name='EROSION_RADIUS'),
#                        Integer(5, 30, "identity", name='MAX_DISTANCE_FOR_ADJACENT_BONES')
#                     ]
#          }
spaces = {
           "GCSegm": [ Real(1., 3., "uniform", name='sigmaSmallScale'),
                       Integer(2, 10, "identity", name='sigmasLargeScale_number'),
                       Real(0.1, 2.0, "uniform", name='sigmasLargeScale_min'),
                       Real(0.15, 0.85, "uniform", name='sigmasLargeScale_step'),
                       Integer(25, 510, "identity", name='lowerThreshold'),
                       Integer(630, 950, "identity", name='upperThreshold'),
                       Real(0.05, 0.9, "uniform", name='binThres_criteria'),
                       Integer(350, 790, "identity", name='bone_criteriaA'),
                       Integer(105, 299, "identity", name='bone_criteriaB'),
                       Real(0.1, 0.8, "uniform", name='bone_smScale_criteria'),
                       # Integer(0, 29, "identity", name='autoROI_criteriaLow'),
                       Integer(30, 55, "identity", name='autoROI_criteriaHigh'),
                       Real(0.15, 1.92, "uniform", name='m_DiffusionFilter_sigma'),
                       Real(2, 26.5, "uniform", name='m_SubstractFilter_scale'),
                       Categorical([3,5,7,9], name='EROSION_RADIUS'),
                       Integer(5, 27, "identity", name='MAX_DISTANCE_FOR_ADJACENT_BONES')
                    ]
         }
autoROI_criteriaLow = 0
# np.min(np.matrix(x_iters[y_iters<1e3]).T[5])
# np.max(np.matrix(x_iters[y_iters<1e3]).T[5])

#%%

def run_optimization(space_key, old_skf, n_calls, n_random_starts, outfile, init_seed):
  # Set hyper-parameters space
  space  = spaces[space_key]
  # Set relevant variables
  params_clsf = all_arguments[space_key]
  ######################################
  # Set objective function to minimize #
  ######################################
  @use_named_args(space)
  def objective(**params):
    global inputCT
    global Inputmetadata
    global ManualSegm
    global bin_presence
    print(params)

    Segm_res = GCAutoSegm(sigmaSmallScale = params['sigmaSmallScale'],
                          sigmasLargeScale_number = params['sigmasLargeScale_number'],
                          sigmasLargeScale_min = params['sigmasLargeScale_min'],
                          sigmasLargeScale_step = params['sigmasLargeScale_step'],
                          lowerThreshold = params['lowerThreshold'],
                          upperThreshold = params['upperThreshold'],
                          binThres_criteria = params['binThres_criteria'],
                          bone_criteriaA = params['bone_criteriaA'],
                          bone_criteriaB = params['bone_criteriaB'],
                          bone_smScale_criteria = params['bone_smScale_criteria'],
                          autoROI_criteriaLow = 0, # params['autoROI_criteriaLow'],
                          autoROI_criteriaHigh = params['autoROI_criteriaHigh'],
                          m_DiffusionFilter_sigma = params['m_DiffusionFilter_sigma'],
                          m_SubstractFilter_scale = params['m_SubstractFilter_scale'],
                          EROSION_RADIUS = np.int(params['EROSION_RADIUS']),
                          MAX_DISTANCE_FOR_ADJACENT_BONES = params['MAX_DISTANCE_FOR_ADJACENT_BONES']
                         )
    # Segm_res = GCAutoSegm(sigmaSmallScale = 1.5,
    #                       sigmasLargeScale_number = 1,
    #                       sigmasLargeScale_min = 0.6,
    #                       sigmasLargeScale_step = 0.2,
    #                       lowerThreshold = 25,
    #                       upperThreshold = 600,
    #                       binThres_criteria = 0.05,
    #                       bone_criteriaA = 400,
    #                       bone_criteriaB = 250,
    #                       bone_smScale_criteria = 0.6,
    #                       autoROI_criteriaLow  = 0,
    #                       autoROI_criteriaHigh = 30,
    #                       m_DiffusionFilter_sigma = 1.0,
    #                       m_SubstractFilter_scale = 10.,
    #                       EROSION_RADIUS = 3,
    #                       MAX_DISTANCE_FOR_ADJACENT_BONES = 15)
    # Segm_res = GCAutoSegm(sigmaSmallScale,
    #                       sigmasLargeScale_number,
    #                       sigmasLargeScale_min,
    #                       sigmasLargeScale_step,
    #                       lowerThreshold,
    #                       upperThreshold,
    #                       binThres_criteria,
    #                       bone_criteriaA,
    #                       bone_criteriaB,
    #                       bone_smScale_criteria,
    #                       autoROI_criteriaLow,
    #                       autoROI_criteriaHigh,
    #                       m_DiffusionFilter_sigma,
    #                       m_SubstractFilter_scale,
    #                       EROSION_RADIUS,
    #                       MAX_DISTANCE_FOR_ADJACENT_BONES)
    # showSome(Segm_res,400)
    GCSegm_arr = itk.GetArrayFromImage(Segm_res)
    all_labels = {i: 0 for i in np.unique(GCSegm_arr) if i>0}
    if len(all_labels.keys()) <2:
      return 1e8
    for z in range(0, GCSegm_arr.shape[0]):
      for i in np.unique(GCSegm_arr[z,:,:]):
        if i>0:
          all_labels[i] +=1
    two_femur = list({k: v for k, v in sorted(all_labels.items(), key=lambda item: item[1], reverse=True)}.keys())[0:2]
    cond_res = (GCSegm_arr != two_femur[0]) & (GCSegm_arr != two_femur[1])
    if np.any(cond_res):
      GCSegm_arr[cond_res] = 0
    GCSegm_arr[GCSegm_arr>0] = 1
    GCSegm = itk.GetImageFromArray(GCSegm_arr.astype(np.int16))
    # showSome(GCSegm,400)
    GCcontours = itk.GetImageFromArray(GetBoundaries(GCSegm, ShortImageType, 0, 1).astype(np.float32))
    GCcontours.SetOrigin(inputCT.GetOrigin())
    GCcontours.SetSpacing(inputCT.GetSpacing())
    GCcontours.SetDirection(inputCT.GetDirection())
    LDMap_finalNoBounds = LDMap(Input1=GCcontours, Input2=Manualcontours, ImageType=FloatImageType)
    # plb.imshow(np.sqrt(LDMap_finalNoBounds[400,:,:]))
    hd_slices = []
    for z in range(LDMap_finalNoBounds[np.logical_not(bin_presence),:,:].shape[0]):
      hd_slices += [np.max(LDMap_finalNoBounds[z,:,:])]
    # plb.plot(np.arange(0,len(hd_slices)), hd_slices)
    res = np.sum(hd_slices)
    # res = np.min(hd_slices)*(np.max(hd_slices) - np.min(hd_slices))
    res += np.sum(LDMap_finalNoBounds[bin_presence,:,:])#!=0
    return res
  ####################
  # Run minimization #
  ####################
  checkpoint_callback = callbacks.CheckpointSaver(outfile, store_objective=False)
  if not old_skf == "":
    print("Retrieving old result and carry on the optimization from where it was left")
    # Reload old skopt object to carry on the optimization
    old_clsf_gp = skload(old_skf)
    args = deepcopy(old_clsf_gp.specs['args'])
    args['n_calls'] += n_calls
    iters   = list(old_clsf_gp.x_iters)
    y_iters = list(old_clsf_gp.func_vals)
    if(isinstance(args['random_state'], np.random.RandomState)):
      args['random_state'] = check_random_state(init_seed)
      # gp_minimize related
      if(isinstance(old_clsf_gp.specs['args']['base_estimator'], GaussianProcessRegressor)):
        args['random_state'].randint(0, np.iinfo(np.int32).max)
    # Define support function for objective
    def check_or_opt(params):
      if(len(iters) > 0):
        y = y_iters.pop(0)
        if(params != iters.pop(0)):
          warnings.warn("Deviated from expected value, re-evaluating", RuntimeWarning)
        else:
          return y
      return objective(params)
    args['callback'] = [checkpoint_callback]
    args['func'] = check_or_opt
    clsf_gp = base_minimize(**args)
    clsf_gp.specs['args']['func'] = objective
  else:
    print("Running minimization from scratch.")
    clsf_gp = gp_minimize(objective,
                          space,
                          acq_func='EI',
                          callback=[checkpoint_callback], # save temporary results
                          n_calls=n_calls,
                          n_random_starts=n_random_starts,
                          random_state=init_seed,
                          noise=1e-10)
  return clsf_gp


#%%


if __name__ == "__main__":

  # Multiple classifiers settings
  global inputCT
  global ManualSegm
  global Inputmetadata
  global bin_presence

  DicomDir = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_CTData"
  ManualSegm_file = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012-label.nrrd"

  FType = itk.ctype('float')
  FloatImageType = itk.Image[FType,3]
  ShortType = itk.ctype('short')
  ShortImageType = itk.Image[ShortType,3]
  # Read Unsupervised and Supervised segmentations
  inputCT, Inputmetadata = dicomsTo3D(DicomDir, ShortType)

  ManualSegm, ManualSegm_metadata = readNRRD(ManualSegm_file, ShortType)
  Manualcontours = itk.GetImageFromArray(GetBoundaries(ManualSegm, ShortImageType, 0, 1).astype(np.float32))
  Manualcontours.SetOrigin(inputCT.GetOrigin())
  Manualcontours.SetSpacing(inputCT.GetSpacing())
  Manualcontours.SetDirection(inputCT.GetDirection())
  # showSome(Manualcontours,ShowIdx)
  Manual_arr = itk.GetArrayFromImage(ManualSegm)
  bin_presence = [ np.sum(Manual_arr[z,:,:])==0 for z in range(Manual_arr.shape[0])]
  # plb.plot(np.arange(0,len(hd_slices)), bin_presence)

  old_results = ""
  n_random_starts = 200
  n_calls = n_random_starts*5
  outfile = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_20201002.pkl"
  old_results = outfile

  # old_results = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_20200923.pkl"
  # Run optimization function
  result = run_optimization(space_key         = "GCSegm",
                            old_skf           = old_results,
                            n_calls           = n_calls,
                            n_random_starts   = n_random_starts,
                            outfile           = outfile,
                            init_seed         = 111)
  # Save final results
  skdump(result, outfile, store_objective=False)


#%%
############
# ANALYSIS #
############
# old_skf = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_20200926.pkl"
old_skf = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_20201002.pkl"
old_clsf_gp = skload(old_skf)
y_iters = np.array(list(old_clsf_gp.func_vals))
len(y_iters)

x_iters = np.array(old_clsf_gp.x_iters)

y_iters_filt = y_iters[y_iters!=1e8]
plb.plot(np.arange(0, len(y_iters_filt)), y_iters_filt)
# plb.plot(np.arange(0, len(old_clsf_gp.x_iters)), y_iters)
# _=plb.hist(y_iters_filt,100)
np.min(y_iters)

sorted(y_iters)

for ag, v in zip(all_arguments['GCSegm'], x_iters[-1]):
  print( " : ".join([ag, np.str(v)]) )

for ag, v in zip(all_arguments['GCSegm'], x_iters[y_iters==sorted(y_iters)[1]][0]):
  print( " : ".join([ag, np.str(v)]) )



for ag, v in zip(all_arguments['GCSegm'], x_iters[np.argmin(y_iters)]):
  print( " : ".join([ag, np.str(v)]) )

#%%
sigmaSmallScale = 1.5
sigmasLargeScale_number = 1
sigmasLargeScale_min = 0.6
sigmasLargeScale_step = 0.2
lowerThreshold = 25
upperThreshold = 600
binThres_criteria = 0.05
bone_criteriaA = 400
bone_criteriaB = 250
bone_smScale_criteria = 0.6
autoROI_criteriaLow  = 0
autoROI_criteriaHigh = 30
m_DiffusionFilter_sigma = 1.0
m_SubstractFilter_scale = 10.
EROSION_RADIUS = 3
MAX_DISTANCE_FOR_ADJACENT_BONES = 15




#%%
Xidx = -2
sigmaSmallScale = x_iters[Xidx][0]
sigmasLargeScale_number = np.int(x_iters[Xidx][1])
sigmasLargeScale_min = x_iters[Xidx][2]
sigmasLargeScale_step = x_iters[Xidx][3]
lowerThreshold = np.int(x_iters[Xidx][4])
upperThreshold = np.int(x_iters[Xidx][5])
binThres_criteria = x_iters[Xidx][6]
bone_criteriaA = np.int(x_iters[Xidx][7])
bone_criteriaB = np.int(x_iters[Xidx][8])
bone_smScale_criteria = x_iters[Xidx][9]
autoROI_criteriaLow = np.int(x_iters[Xidx][10])
autoROI_criteriaHigh = np.int(x_iters[Xidx][11])
m_DiffusionFilter_sigma = x_iters[Xidx][12]
m_SubstractFilter_scale = x_iters[Xidx][13]
EROSION_RADIUS = np.int(x_iters[Xidx][14])
MAX_DISTANCE_FOR_ADJACENT_BONES = np.int(x_iters[Xidx][15])
#%%

#%%
# sigmaSmallScale = x_iters[np.argmin(y_iters)][0]
# sigmasLargeScale_number = np.int(x_iters[np.argmin(y_iters)][1])
# sigmasLargeScale_min = x_iters[np.argmin(y_iters)][2]
# sigmasLargeScale_step = x_iters[np.argmin(y_iters)][3]
# lowerThreshold = np.int(x_iters[np.argmin(y_iters)][4])
# upperThreshold = np.int(x_iters[np.argmin(y_iters)][5])
# binThres_criteria = x_iters[np.argmin(y_iters)][6]
# bone_criteriaA = np.int(x_iters[np.argmin(y_iters)][7])
# bone_criteriaB = np.int(x_iters[np.argmin(y_iters)][8])
# bone_smScale_criteria = x_iters[np.argmin(y_iters)][9]
# autoROI_criteriaLow = np.int(x_iters[np.argmin(y_iters)][10])
# autoROI_criteriaHigh = np.int(x_iters[np.argmin(y_iters)][11])
# m_DiffusionFilter_sigma = x_iters[np.argmin(y_iters)][12]
# m_SubstractFilter_scale = x_iters[np.argmin(y_iters)][13]
# EROSION_RADIUS = np.int(x_iters[np.argmin(y_iters)][14])
# MAX_DISTANCE_FOR_ADJACENT_BONES = np.int(x_iters[np.argmin(y_iters)][15])

#%%
sigmaSmallScale = x_iters[np.argmin(y_iters)][0]
sigmasLargeScale_number = np.int(x_iters[np.argmin(y_iters)][1])
sigmasLargeScale_min = x_iters[np.argmin(y_iters)][2]
sigmasLargeScale_step = x_iters[np.argmin(y_iters)][3]
lowerThreshold = np.int(x_iters[np.argmin(y_iters)][4])
upperThreshold = np.int(x_iters[np.argmin(y_iters)][5])
binThres_criteria = x_iters[np.argmin(y_iters)][6]
bone_criteriaA = np.int(x_iters[np.argmin(y_iters)][7])
bone_criteriaB = np.int(x_iters[np.argmin(y_iters)][8])
bone_smScale_criteria = x_iters[np.argmin(y_iters)][9]
autoROI_criteriaHigh = np.int(x_iters[np.argmin(y_iters)][10])
m_DiffusionFilter_sigma = x_iters[np.argmin(y_iters)][11]
m_SubstractFilter_scale = x_iters[np.argmin(y_iters)][12]
EROSION_RADIUS = np.int(x_iters[np.argmin(y_iters)][13]) # 5
MAX_DISTANCE_FOR_ADJACENT_BONES = np.int(x_iters[np.argmin(y_iters)][14]) # 18



#%%

# Quality best till now
sigmaSmallScale = 2.3431706848114375
sigmasLargeScale_number = 7
sigmasLargeScale_min = 0.5611119614464578
sigmasLargeScale_step = 0.2062579541371359
lowerThreshold = 25
upperThreshold = 600
binThres_criteria = 0.05
bone_criteriaA = 400
bone_criteriaB = 250
bone_smScale_criteria = 0.5080780340864003
autoROI_criteriaLow  = 0
autoROI_criteriaHigh = 30
m_DiffusionFilter_sigma = 0.8316166698338118
m_SubstractFilter_scale = 10.
EROSION_RADIUS = 3
MAX_DISTANCE_FOR_ADJACENT_BONES = 15
# res = 135.03032

#%%
Segm_res = GCAutoSegm(sigmaSmallScale,
                      sigmasLargeScale_number,
                      sigmasLargeScale_min,
                      sigmasLargeScale_step,
                      lowerThreshold,
                      upperThreshold,
                      binThres_criteria,
                      bone_criteriaA,
                      bone_criteriaB,
                      bone_smScale_criteria,
                      autoROI_criteriaLow,
                      autoROI_criteriaHigh,
                      m_DiffusionFilter_sigma,
                      m_SubstractFilter_scale,
                      EROSION_RADIUS,
                      MAX_DISTANCE_FOR_ADJACENT_BONES
                     )
GCSegm_arr = itk.GetArrayFromImage(Segm_res)
all_labels = {i: 0 for i in np.unique(GCSegm_arr) if i>0}
if len(all_labels.keys()) <2:
  print("Nope")
for z in range(0, GCSegm_arr.shape[0]):
  for i in np.unique(GCSegm_arr[z,:,:]):
    if i>0:
      all_labels[i] +=1
two_femur = list({k: v for k, v in sorted(all_labels.items(), key=lambda item: item[1], reverse=True)}.keys())[0:2]
cond_res = (GCSegm_arr != two_femur[0]) & (GCSegm_arr != two_femur[1])
if np.any(cond_res):
  GCSegm_arr[cond_res] = 0
GCSegm_arr[GCSegm_arr>0] = 1
GCSegm = itk.GetImageFromArray(GCSegm_arr.astype(np.int16))
# showSome(GCSegm,400)
# GCcontours = itk.GetImageFromArray(GetBoundaries(GCSegm, ShortImageType, 0, 1).astype(np.float32))
# GCcontours.SetOrigin(inputCT.GetOrigin())
# GCcontours.SetSpacing(inputCT.GetSpacing())
# GCcontours.SetDirection(inputCT.GetDirection())
# showSome(GCcontours,400)
# showSome(ManualSegm,400)
LDMap_finalNoBounds = LDMap(Input1=castImage(GCSegm, FloatImageType), Input2=castImage(ManualSegm, FloatImageType), ImageType=FloatImageType)
# plb.imshow(np.sqrt(LDMap_finalNoBounds[400,:,:]))
hd_slices = []
for z in range(LDMap_finalNoBounds[np.logical_not(bin_presence),:,:].shape[0]):
  hd_slices += [np.max(LDMap_finalNoBounds[z,:,:])]
# plb.plot(np.arange(0,len(hd_slices)), hd_slices)
res = np.min(hd_slices)*(np.max(hd_slices) - np.min(hd_slices))
res += np.sum(LDMap_finalNoBounds[bin_presence,:,:])!=0
print(res)


#%%
# all_originals = os.listdir("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/original/")
# for s in all_originals:
#   print(s)
#   DicomDir_temp = os.path.join("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/original/", s)
#   inputCT, Inputmetadata = dicomsTo3D(DicomDir_temp, ShortType)
#   # showSome(inputCT, )
#   Segm_res = GCAutoSegm(sigmaSmallScale,
#                         sigmasLargeScale_number,
#                         sigmasLargeScale_min,
#                         sigmasLargeScale_step,
#                         lowerThreshold,
#                         upperThreshold,
#                         binThres_criteria,
#                         bone_criteriaA,
#                         bone_criteriaB,
#                         bone_smScale_criteria,
#                         autoROI_criteriaLow,
#                         autoROI_criteriaHigh,
#                         m_DiffusionFilter_sigma,
#                         m_SubstractFilter_scale,
#                         EROSION_RADIUS,
#                         MAX_DISTANCE_FOR_ADJACENT_BONES
#                        )
#   outdir = os.path.join("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/GCSegm_14102020", s)
#   os.mkdir(outdir)
#   Volume3DToDicom(imgObj = Segm_res,
#                   MetadataObj = Inputmetadata,
#                   outdir = outdir)
#   print("Done\n\n\n")

#%%
def GetFemurs(Segm_res):
  GCSegm_arr = itk.GetArrayFromImage(Segm_res)
  all_labels = {i: 0 for i in np.unique(GCSegm_arr) if i>0}
  if len(all_labels.keys()) <2:
    print("Nope")
  for z in range(0, GCSegm_arr.shape[0]):
    for i in np.unique(GCSegm_arr[z,:,:]):
      if i>0:
        all_labels[i] +=1
  two_femur = list({k: v for k, v in sorted(all_labels.items(), key=lambda item: item[1], reverse=True)}.keys())[0:2]
  cond_res = (GCSegm_arr != two_femur[0]) & (GCSegm_arr != two_femur[1])
  if np.any(cond_res):
    GCSegm_arr[cond_res] = 0
  GCSegm_arr[GCSegm_arr>0] = 1
  GCSegm = itk.GetImageFromArray(GCSegm_arr.astype(np.int16))
  return GCSegm
#%%
all_originals = os.listdir("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/original/")
s = all_originals[12]
for s in all_originals:
  print(s)
  DicomDir_temp = os.path.join("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/original/", s)
  DicomDir = DicomDir_temp
  # outdir = os.path.join("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/GCSegm_30092020", s)
  outdir = os.path.join("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/GCSegm_14102020", s)
  manual_seg = os.path.join("/home/PERSONALE/daniele.dallolio3/femur_segmentation/AllData/manual_segmentation", "".join([s, "-label.nrrd"]))
  inputCT, Inputmetadata = dicomsTo3D(DicomDir_temp, ShortType)
  showSome(inputCT, 225)
  outputCT_, Outputmetadata = dicomsTo3D(outdir, ShortType)
  outputCT = GetFemurs(outputCT_)
  ManualCT, Manualmetadata = readNRRD(manual_seg, ShortType)
  ShowIdx = 390
  showSome(inputCT, ShowIdx)
  showSome(outputCT, ShowIdx)
  showSome(ManualCT, ShowIdx)

#%%


#######
# END #
#######
