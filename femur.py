import itk
import numpy as np
import ctypes
import matplotlib.pylab as plb
from scipy.ndimage.morphology import distance_transform_cdt
import os
import sys
sys.path.append('/home/PERSONALE/daniele.dallolio3/3DFemurSegmentation/lib')
#%%
import fastDistMatrix
import GraphCutSupport
#%%

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
  dictionary_keys = filter(lambda x: not "ITK_" in x, dicom_reader.GetMetaDataDictionary().GetKeys())
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
  # al3_null = eigenvalues_matrix[:,:,:,2] > threshold
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
# plb.hist(cpp_rs[4].to_numpy()-RsImage[:,:,:,0].flatten())


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
    # Sort them by abs
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
# RsImg = np.zeros(RsImg.shape)
# RsImg[:,:,:,0] = cpp_rs[4].to_numpy().reshape(EigsImg.shape[:-1])
# RsImg[:,:,:,1] = cpp_rs[5].to_numpy().reshape(EigsImg.shape[:-1])
# RsImg[:,:,:,2] = cpp_rs[6].to_numpy().reshape(EigsImg.shape[:-1])
# RsImg[:,:,:,3] = cpp_noise[3].to_numpy().reshape(EigsImg.shape[:-1])
# EigsImg[:,:,:,0] = cpp_eigs[3].to_numpy().reshape(HessianMatrices[0].shape)
# EigsImg[:,:,:,1] = cpp_eigs[4].to_numpy().reshape(HessianMatrices[0].shape)
# EigsImg[:,:,:,2] = cpp_eigs[5].to_numpy().reshape(HessianMatrices[0].shape)
# plb.hist(cpp_meas[3].to_numpy()-SheetnessImage.flatten())





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


# M11, M12, M13, M22, M23, M33 = ((cpp_hxx[3]).to_numpy().reshape(hxx.shape),(cpp_hxy[3]).to_numpy().reshape(hxx.shape),(cpp_hxz[3]).to_numpy().reshape(hxx.shape),(cpp_hyy[3]).to_numpy().reshape(hxx.shape),(cpp_hyz[3]).to_numpy().reshape(hxx.shape),(cpp_hzz[3]).to_numpy().reshape(hxx.shape))
# M11, M12, M13, M22, M23, M33=hxx, hxy, hxz, hyy, hyz, hzz
def GetEigenValues(M11, M12, M13, M22, M23, M33, roi=None):
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
  EigenValues = np.zeros(M11.shape + (4,))
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
  # SRImg.tolist() == itk.GetArrayFromImage(cpp_SmoothingA).tolist()
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
# SRInput=cpp_RightBeforeSmoothingA
# SRInput= itk.GetImageFromArray(itk.GetArrayFromImage(cpp_RightBeforeSmoothingA))
# itk.GetArrayFromImage(cpp_SmoothingA).tolist()==itk.GetArrayFromImage(output_image).tolist()
# prova=itk.GetArrayFromImage(cpp_SmoothingA)!=itk.GetArrayFromImage(output_image)
# np.min(itk.GetArrayFromImage(cpp_SmoothingA)[prova]-itk.GetArrayFromImage(output_image)[prova])
# np.max(itk.GetArrayFromImage(cpp_SmoothingA)[prova]-itk.GetArrayFromImage(output_image)[prova])


def singlescaleSheetness(singleScaleInput,
                         scale,
                         SmoothingImageType,
                         roi = None,
                         alpha = 0.5,
                         beta = 0.5,
                         gamma = 0.5):
  print("Computing single-scale sheetness, sigma=%4.2f"% scale)
  # np.unique(itk.GetArrayFromImage(singleScaleInput)==itk.GetArrayFromImage(cpp_RightBeforeSmoothingA).tolist())
  SmoothImg = SmoothingRecursive(SRInput = singleScaleInput,
                                 sigma = scale,
                                 SRImageType = SmoothingImageType)
  # itk.GetArrayFromImage(SmoothImg).tolist() == itk.GetArrayFromImage(cpp_SmoothingA).tolist()
  HessianMatrices = computeQuasiHessian(SmoothImg)
  # hxx, hxy, hxz, hyy, hyz, hzz = HessianMatrices
  # np.max(cpp_hxx[3]-hxx.flatten())
  # np.max(cpp_hxy[3]-hxy.flatten())
  # np.max(cpp_hxz[3]-hxz.flatten())
  # np.max(cpp_hyy[3]-hyy.flatten())
  # np.max(cpp_hyz[3]-hyz.flatten())
  # np.max(cpp_hzz[3]-hzz.flatten())
  EigenImg = GetEigenValues(*HessianMatrices, roi)
  # EigenImg = np.zeros(M11.shape + (4,))
  # EigenImg[:,:,:,0] = cpp_eigs[3].to_numpy().reshape(HessianMatrices[0].shape)
  # EigenImg[:,:,:,1] = cpp_eigs[4].to_numpy().reshape(HessianMatrices[0].shape)
  # EigenImg[:,:,:,2] = cpp_eigs[5].to_numpy().reshape(HessianMatrices[0].shape)

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
  # itk.GetArrayFromImage(multiscaleSheetness).tolist()==itk.GetArrayFromImage(cpp_smallScaleSheetnessImage).tolist()
  # prova=itk.GetArrayFromImage(multiscaleSheetness)!=itk.GetArrayFromImage(cpp_smallScaleSheetnessImage)
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


def DistanceTransform(ChamferInput):
  distanceMap = np.zeros(ChamferInput.shape)
  _infinityDistance = np.sum(ChamferInput.shape) + 1
  distanceMap[ChamferInput == 0] = _infinityDistance
  distanceMapPad = np.pad(distanceMap, 1, mode='constant', constant_values=(_infinityDistance, _infinityDistance))
  distanceMap = fastDistMatrix.ManhattanChamferDistance(distanceMapPad, distanceMap.shape)
  distanceMap = distanceMap[1:-1, 1:-1, 1:-1].copy()
  return distanceMap

#%%
########################
# Segmentation section #
########################

# SheetnessBasedSmoothCost_compute:
def SheetnessBasedSmoothCost(pixelLeft,
                             pixelRight,
                             shtnLeft,
                             shtnRight):
  COST_AMPLIFIER = 1000
  alpha = 5.0
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
                 ROI):
  COST_AMPLIFIER = 1000
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
                                                              shtnRight = sheetness[:,:,1:])
  Ycenters, YFromCenter, YToCenter = SheetnessBasedSmoothCost(pixelLeft  = _pixelIdImage[:, :-1, :],
                                                              pixelRight = _pixelIdImage[:, 1:, :],
                                                              shtnLeft  = sheetness[:,:-1,:],
                                                              shtnRight = sheetness[:,1:,:])
  Zcenters, ZFromCenter, ZToCenter = SheetnessBasedSmoothCost(pixelLeft  = _pixelIdImage[:-1,:,:],
                                                              pixelRight = _pixelIdImage[1:,:,:],
                                                              shtnLeft  = sheetness[:-1,:,:],
                                                              shtnRight = sheetness[1:,:,:])
  CentersPixels = np.concatenate([Zcenters[0], Ycenters[0], Xcenters[0] ])
  NeighborsPixels = np.concatenate([Zcenters[1], Ycenters[1], Xcenters[1] ])
  _totalNeighbors = len(NeighborsPixels)
  flat_smoothCostFromCenter = np.concatenate([ZFromCenter, YFromCenter, XFromCenter ])
  flat_smoothCostToCenter = np.concatenate([ZToCenter, YToCenter, XToCenter ])

  # sorted_idxs= np.lexsort((NeighborsPixels, CentersPixels))
  # CentersPixels = CentersPixels[sorted_idxs]
  # NeighborsPixels = NeighborsPixels[sorted_idxs]
  # flat_smoothCostFromCenter = flat_smoothCostFromCenter[sorted_idxs]
  # flat_smoothCostToCenter = flat_smoothCostToCenter[sorted_idxs]
  #
  # CentersPixels.tolist()==cpp_NeighInfo.iloc[:,0].tolist()
  # NeighborsPixels.tolist()==cpp_NeighInfo.iloc[:,1].tolist()
  # flat_smoothCostFromCenter.tolist()==cpp_NeighInfo.iloc[:,2].tolist()
  # flat_smoothCostToCenter.tolist()==cpp_NeighInfo.iloc[:,3].tolist()

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

  # check_roi = np.array(np.where(roi!=0)).T
  # np.unique(_pixelIdImage[roi!=0] == cpp_dataCostInfo[3])
  # np.unique(cpp_dataCostInfo[0].apply(lambda x: int(x.replace("[","")))==check_roi[:,2])
  # np.unique(cpp_dataCostInfo[1]==check_roi[:,1])
  # np.unique(cpp_dataCostInfo[2].apply(lambda x: int(x.replace("]","")))==check_roi[:,0])
  # cpp_dataCostInfo[2].apply(lambda x: int(x.replace("]","")))
  # cpp_dataCostInfo
  # np.unique(flat_dataCostSource == cpp_dataCostInfo[4])
  # np.unique(flat_dataCostSink == cpp_dataCostInfo[5])


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
  # len(flat_smoothCostFromCenter)
  # organize_scf = pd.DataFrame(data=[CentersPixels, NeighborsPixels, flat_smoothCostFromCenter, flat_smoothCostToCenter]).T
  # organize_scf = organize_scf.sort_values(by=[0,1])
  # organize_scf.iloc[:,0].tolist()==cpp_NeighInfo.iloc[:,0].tolist()
  # organize_scf.iloc[:,1].tolist()==cpp_NeighInfo.iloc[:,1].tolist()
  # organize_scf.iloc[:,2].tolist()==cpp_NeighInfo.iloc[:,2].tolist()
  # organize_scf.iloc[:,3].tolist()==cpp_NeighInfo.iloc[:,3].tolist()

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
  # itk.GetArrayFromImage(gcresult).tolist()==itk.GetArrayFromImage(cpp_gcOutput).tolist()
  # _labelIdImage.tolist()==itk.GetArrayFromImage(cpp_gcOutput).tolist()
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

# plb.hist(np.unique(itk.GetArrayFromImage(fastMarcher.GetOutput())))
# len(TrialIndexes)
# prova = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/provaA.txt", header=None)
# np.unique(prova[prova[0]==27][3].apply(lambda x: int(x.replace("]","")))==TrialIndexes[:,0])
# np.unique(prova[prova[0]==27][2]==TrialIndexes[:,1])
# np.unique(prova[prova[0]==27][1].apply(lambda x: int(x.replace(" [","")))==TrialIndexes[:,2])


#%%
import matplotlib.pylab as plb
def showSome(imgObj, idx = 0):
  prova = itk.GetArrayFromImage(imgObj)
  plb.imshow(prova[idx,:,:])

def Read3DNifti(fn, t = 'unsigned char'):
  nifti_obj = itk.NiftiImageIO.New()
  set_type = itk.ctype(t)
  reader_type = itk.Image[set_type,3]
  reader = itk.ImageFileReader[reader_type].New()
  reader.SetFileName(fn)
  reader.SetImageIO(nifti_obj)
  reader.Update()
  return reader.GetOutput()

def duplicate(img,
              ImageType):
  DuplicatorType = itk.ImageDuplicator[ImageType]
  duplicator = DuplicatorType.New()
  duplicator.SetInputImage( img )
  duplicator.Update()
  return duplicator.GetOutput()

#%%


if __name__ == "__main__":
#%%
  #
  # Add parser
  #
  #

  # Parameters
  sigmaSmallScale = 1.5
  sigmasLargeScale = [0.6, 0.8]
  lowerThreshold = 25
  upperThreshold = 600

  DicomDir = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/CT_femore_Viceconti/CT_dataset_FemoreEsempio/"
  # DicomDir = "/home/daniele/OneDrive/Lavori/CT_femore_Viceconti/CT_dataset_FemoreEsempio/"

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

  # Read Dicoms Series and turn it into 3D object
  inputCT, Inputmetadata = dicomsTo3D(DicomDir, ShortType)
  showSome(inputCT,50)

  print("Preprocessing")
#%%

  # Preprocessing
  print("Thresholding input image")
  thresholdedInputCT = thresholding(duplicate(inputCT, ShortImageType), lowerThreshold, upperThreshold) # Checked

  showSome(thresholdedInputCT, 50)
  smallScaleSheetnessImage = multiscaleSheetness(multiScaleInput = castImage(thresholdedInputCT, OutputType=FloatImageType),
                                                 scales = [sigmaSmallScale],
                                                 SmoothingImageType = FloatImageType)

  showSome(smallScaleSheetnessImage,50)
  # smallScaleSheetnessImage = cpp_smallScaleSheetnessImage
  print("Estimating soft-tissue voxels")
  smScale_bin = binaryThresholding(inputImage = smallScaleSheetnessImage,
                                   lowerThreshold = -0.05,
                                   upperThreshold = 0.05,
                                   outputImageType = UCType)
  showSome(smScale_bin,50)
  # itk.GetArrayFromImage(smScale_bin).tolist()==itk.GetArrayFromImage(cpp_thresA).tolist()
  smScale_cc = ConnectedComponents(inputImage = smScale_bin,
                                   outputImageType = itk.ctype('unsigned long')
                                  )
  showSome(smScale_cc,50)
  # itk.GetArrayFromImage(castImage(smScale_cc,OutputType=UCImageType)).tolist()==itk.GetArrayFromImage(cpp_cc).tolist()
  # smScale_cc=cpp_cc
  smScale_rc = RelabelComponents(inputImage = smScale_cc,
                                 outputImageType=None)
  showSome(smScale_rc,50)
  # itk.GetArrayFromImage(smScale_rc).tolist()==itk.GetArrayFromImage(cpp_rc).astype(np.uint64).tolist()
  # smScale_rc=cpp_rc
  softTissueEstimation = binaryThresholding(inputImage = smScale_rc,
                                            lowerThreshold = 1,
                                            upperThreshold = 1)
  # itk.GetArrayFromImage(softTissueEstimation).tolist()==itk.GetArrayFromImage(cpp_soft).tolist()
  showSome(softTissueEstimation, 50)

  print("Estimating bone voxels")
  boneEstimation = itk.GetArrayFromImage(inputCT)
  smScale = itk.GetArrayFromImage(smallScaleSheetnessImage)
  boneCondition = (boneEstimation > 400) | (boneEstimation > 250) & (smScale > 0.6)
  boneEstimation[boneCondition] = 1
  boneEstimation[np.logical_not(boneCondition)] = 0
  print("Computing ROI from bone estimation using Chamfer Distance")
  # boneDist = distance_transform_cdt(boneEstimation.astype(np.int64),
  #                                   metric='taxicab',
  #                                   return_distances=True).astype(np.float64)
  boneDist = DistanceTransform(boneEstimation.astype(np.int32)).astype(np.float32)
  boneDist = itk.GetImageFromArray(boneDist)
  boneDist.SetOrigin(inputCT.GetOrigin())
  boneDist.SetSpacing(inputCT.GetSpacing())
  boneDist.SetDirection(inputCT.GetDirection())
  showSome(boneDist, 50)
  autoROI  = binaryThresholding(inputImage = boneDist,
                                lowerThreshold = 0,
                                upperThreshold = 30,
                                outputImageType = UCType)
  showSome(autoROI, 50)

  print("Unsharp masking")
  InputCT_float = castImage(inputCT, OutputType=FloatImageType)
  # itk.GetArrayFromImage(InputCT_float).tolist()==itk.GetArrayFromImage(cpp_inputCTfloat).tolist()

  # I*G (discrete gauss)
  m_DiffusionFilter = Gaussian(GaussInput = InputCT_float,
                               sigma = 1.0)
  # itk.GetArrayFromImage(m_DiffusionFilter).tolist()==itk.GetArrayFromImage(cpp_m_DiffusionFilter).tolist()
  showSome(m_DiffusionFilter, 50)

  # I - (I*G)
  m_SubstractFilter = substract(InputCT_float, m_DiffusionFilter)
  # itk.GetArrayFromImage(m_SubstractFilter).tolist()==itk.GetArrayFromImage(cpp_m_SubstractFilter).tolist()
  showSome(m_SubstractFilter, 50)

  # k(I-(I*G))
  m_MultiplyFilter = linearTransform(m_SubstractFilter,
                                     scale = 10.,
                                     shift = 0.)
  # itk.GetArrayFromImage(m_MultiplyFilter).tolist()==itk.GetArrayFromImage(cpp_m_MultiplyFilter).tolist()
  showSome(m_MultiplyFilter, 50)

  # I+k*(I-(I*G))
  inputCTUnsharpMasked = add(InputCT_float, m_MultiplyFilter)
  # itk.GetArrayFromImage(inputCTUnsharpMasked).tolist()==itk.GetArrayFromImage(cpp_inputCTUnsharpedMasked).tolist()
  showSome(inputCTUnsharpMasked, 50)

  print("Computing multiscale sheetness measure at %d scales" % len(sigmasLargeScale))
  Sheetness = multiscaleSheetness(multiScaleInput=inputCTUnsharpMasked,
                                  scales = sigmasLargeScale,
                                  SmoothingImageType = FloatImageType,
                                  roi = None)
  showSome(Sheetness, 50)
  # plb.hist(itk.GetArrayFromImage(Sheetness).flatten()-itk.GetArrayFromImage(cpp_sheetness).flatten())

#%%
  ###########
  # Segment #
  ###########
  print("Segmentation")
  gcResult = Segmentation(imgObj = inputCT,
                          softEst = softTissueEstimation,
                          sht = Sheetness,
                          ROI = autoROI
                          )
  showSome(gcResult, 50)
  # showSome(cpp_gc, 50)
  # itk.GetArrayFromImage((gcResult)).tolist() == (itk.GetArrayFromImage(cpp_gc)).tolist()
#%%
  ###################
  # Bone-separation #
  ###################
  print("Bone Separation")
  EROSION_RADIUS = 3
  MAX_DISTANCE_FOR_ADJACENT_BONES = 15

  print("Computing Connected Components")
  mainIslands = ConnectedComponents(inputImage = gcResult,
                                    outputImageType = ULType)
  # itk.GetArrayFromImage((mainIslands)).tolist() == (itk.GetArrayFromImage(cpp_mainIslands)).tolist()
  showSome(mainIslands, 50)
  print("Erosion + Connected Components, ball radius=%d"% EROSION_RADIUS)
  eroded_gc = erosion(gcResult, EROSION_RADIUS, UCType)
  # itk.GetArrayFromImage((eroded_gc)).tolist() == (itk.GetArrayFromImage(cpp_eroded)).tolist()
  showSome(eroded_gc, 50)
  subIslands = ConnectedComponents(inputImage = eroded_gc,
                                   outputImageType = ULType)
  # itk.GetArrayFromImage((subIslands)).tolist() == (itk.GetArrayFromImage(cpp_subIslands)).tolist()
  showSome(subIslands, 50)
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
                                          stoppingValue = MAX_DISTANCE_FOR_ADJACENT_BONES + 1,
                                          ImageType = FloatImageType)
      for jdx in range(idx+1, len(subIslandsSortedBySize[l])):
        potentialAdjacentSubIsland = subIslandsSortedBySize[l][jdx]
        islandsAdjacent = isIslandWithinDistance(image = subIslands,
                                                 distanceImage = distance,
                                                 label = potentialAdjacentSubIsland,
                                                 maxDistance = MAX_DISTANCE_FOR_ADJACENT_BONES
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
    # itk.GetArrayFromImage((gcOutput)).tolist() == (itk.GetArrayFromImage(cpp_gcOutput)).tolist()
    # updateResult
    uniqueLabel = np.max(mainArray) + 1
    gcValues = itk.GetArrayFromImage(gcOutput)
    mainArray[gcValues==1] = uniqueLabel # result
  relabelled_mainIslands = itk.GetImageFromArray(mainArray)
  relabelled_mainIslands.SetOrigin(mainIslands.GetOrigin())
  relabelled_mainIslands.SetSpacing(mainIslands.GetSpacing())
  relabelled_mainIslands.SetDirection(mainIslands.GetDirection())
  finalResult = RelabelComponents(inputImage = relabelled_mainIslands,
                                  outputImageType = UCType)
  # itk.GetArrayFromImage((finalResult)).tolist() == (itk.GetArrayFromImage(cpp_finalResult)).tolist()
  showSome(finalResult, 50)
#%%
  # Save results
  print("Saving")
  print("Writing the result to %s"% outfile)
  # Still choosing what to dump




####################
# WORK IN PROGRESS #
####################
#%%
import pandas as pd
import os
# Check Pre-Processing
# cpp_inputCT = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/PrimoEsempio.nii", t = 'short')
# Ok: cpp_thresholded = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/thresholdedInputCT.nii", t = 'short')
# Ok: cpp_thresholdedCast = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/thresholdedInputCT_short.nii", t = 'float')
# Ok: cpp_RightBeforeSmoothingA = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/RightBeforeSmoothinRecursive1.500000.nii", t= 'float')
# Ok: cpp_SmoothingA = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/SmoothinRecursive1.500000.nii", t= 'float')
# Ok: if e-5 differences taken (floating precision between numpy and c++): cpp_hxx = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/hxx_1.500000.txt", header=None)
# Ok: if e-5 differences taken (floating precision between numpy and c++): cpp_hxy = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/hxy_1.500000.txt", header=None)
# Ok: if e-5 differences taken (floating precision between numpy and c++): cpp_hxz = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/hxz_1.500000.txt", header=None)
# Ok: if e-5 differences taken (floating precision between numpy and c++): cpp_hyy = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/hyy_1.500000.txt", header=None)
# Ok: if e-5 differences taken (floating precision between numpy and c++): cpp_hyz = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/hyz_1.500000.txt", header=None)
# Ok: if e-5 differences taken (floating precision between numpy and c++): cpp_hzz = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/hzz_1.500000.txt", header=None)
# Ok within float precision (few mismatchs due to arccos computations and sorting, they all depend on low precision: e.g. when sorting eigenvalues by abs you can have almost same abs value, only differences at 1e-4 or 1e-5)
# cpp_eigs = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/eigs_1.500000.txt", header=None)
# Ok within float precision throughout computation: cpp_rs = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/rs_1.500000.txt", header=None)
# Ok within float precision(0.0002): cpp_noise = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/noise_1.500000.txt", header=None)
# Ok within float precision throughout computation: cpp_meas = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/meas_1.500000.txt", header=None)
# Ok within float precision throughout computation: cpp_smallScaleSheetnessImage = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/smallScaleSheetnessImage.nii", t= 'float')
# Ok: cpp_thresA = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/smScale_bin.nii")
# Ok: cpp_cc = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/smScale_cc.nii")
# Not Ok but because the ordering of labels with same counts is not the same (does not matter): cpp_rc = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/smScale_rc.nii", "float")
# Ok: cpp_soft = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/softTissueEstimation_.nii")
# Ok: cpp_boneEstimation = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/boneEstimation.nii")
# Ok: cpp_chamferResult = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/chamferResult.nii", t = 'float')
# Ok: cpp_roi = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/roi.nii")
# Ok: cpp_inputCTfloat = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/InputCT_float.nii", t='float')
# Ok: cpp_m_DiffusionFilter = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/m_DiffusionFilter.nii", t='float')
# Ok: cpp_m_SubstractFilter = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/m_SubstractFilter.nii", t='float')
# Ok: cpp_m_MultiplyFilter = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/m_MultiplyFilter.nii", t='float')
# Ok: cpp_inputCTUnsharpedMasked = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/inputCTUnsharpedMasked.nii", t='float')
# Ok within float precision throughout computation: cpp_sheetness = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/sheetnessRightBeforeSegm.nii", t = 'float')

# Check Segmentation
# Starting from:
# cpp_soft = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/soft-tissue-est.nii")
# cpp_sheetness = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/sheetnessRightBeforeSegm.nii", t = 'float')
# cpp_roi = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/roi.nii")
# softTissueEstimation=cpp_soft
# Sheetness=cpp_sheetness
# autoROI=cpp_roi
# Ok: cpp_pixelIdImage = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/_pixelIdImage.nii", t='float')
# Ok: cpp_dataCostInfo = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/initDataCosts.txt", header=None)
# Ok: cpp_NeighInfo = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/initNeigh.txt", header=None)
# Ok: cpp_gc = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/gc-output-part-0.nii")

# Check Bone-separation
# Starting from: cpp_assembled = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/assembledResult.nii")
# Ok: cpp_mainIslands = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/mainIslands.nii")
# Ok: cpp_eroded = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/erodedMain.nii")
# Ok: cpp_subIslands = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/subIslands.nii")
# Ok: cpp_distance = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/distance_28.nii", t = 'float')
# Ok: cpp_roiSub = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/roi_sub_5_1_1.nii")
# Ok: cpp_pixelIdImage = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/_pixelIdImage.nii", t='float')
# Ok: cpp_dataCostInfo = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/initDataCosts.txt", header=None)
# Ok: cpp_NeighInfo = pd.read_csv("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/initNeigh.txt", header=None)
# Ok: cpp_gcOutput = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/gcOut_5_1_1.nii")
# Ok: cpp_updateResult = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/resultAfter_5_1_1.nii")
# Ok: cpp_finalResult = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/finalResult.nii")
# Done

#%%


###########
# THE END #
###########
