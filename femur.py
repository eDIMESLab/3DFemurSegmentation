import itk
import numpy as np
import ctypes
import matplotlib.pylab as plb
from scipy.ndimage.morphology import distance_transform_cdt
#%%
import sys
sys.path.append('/home/PERSONALE/daniele.dallolio3/3DFemurSegmentation')
#%%
import fast_distance_matrix


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
  metadata = [ {tag:fdcm[tag] for tag in dicom_reader.GetMetaDataDictionary().GetKeys()} for fdcm in metadataArray ]
  return imgs_seq, metadata


def thresholding(inputImage,
                 lowerThreshold = 25,
                 upperThreshold = 600):
  np_inputImage = itk.GetArrayFromImage(inputImage)
  np_inputImage = np.minimum( upperThreshold, np.maximum( np_inputImage, lowerThreshold) )
  return itk.GetImageFromArray(np_inputImage)


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
  eigenvalues_matrix[:,:,:,:3] = np.sort(eigenvalues_matrix[:,:,:,:3])
  det_image = np.sum(eigenvalues_matrix, axis=-1)
  if not roi is None:
    inside = roi!=0
    mean_norm = 1. / np.mean(det_image[inside])
  else:
    mean_norm = 1. / np.mean(det_image)
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
    l1, l2, l3 = sortedEigs[:,:,:,0], sortedEigs[:,:,:,1], sortedEigs[:,:,:,2]
    condA = np.abs(l1) > np.abs(l2)
    l1[condA], l2[condA] = l2[condA], l1[condA]
    condB = np.abs(l2) > np.abs(l3)
    l2[condB], l3[condB] = l3[condB], l2[condB]
    condC = np.abs(l1) > np.abs(l2)
    l1[condC], l2[condC] = l2[condC], l1[condC]
    sortedEigs[:,:,:,0], sortedEigs[:,:,:,1], sortedEigs[:,:,:,2] = l1, l2, l3
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
  SRObj.SetSigma(sigma)
  SRObj.Update()
  return SRObj.GetOutput()


def singlescaleSheetness(singleScaleInput,
                         scale,
                         SmoothingImageType,
                         roi = None,
                         alpha = 0.5,
                         beta = 0.5,
                         gamma = 0.5):
  print("Computing single-scale sheetness, sigma=%4.2f"% scale)

  # SmoothImg = SmoothingRecursive(SRInput = singleScaleInput,
  #                                sigma = scale,
  #                                SRImageType = SmoothingImageType)
  # HessianMatrices = computeQuasiHessian(SmoothImg)
  # EigenImg = GetEigenValues(*HessianMatrices, roi)
  SmoothImg = computeHessian(HessInput = singleScaleInput,
                             sigma = scale,
                             HRGImageType = SmoothingImageType)
  EigenImg = computeEigenvalues(SmoothImg = SmoothImg,
                                EigType = itk.ctype('float'),
                                D = 3)
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
  return multiscaleSheetness


def binaryThresholding(inputImage,
                       lowerThreshold,
                       upperThreshold,
                       outputImageType = None,
                       insideValue = 1,
                       outsideValue = 0):
  s,d = itk.template(inputImage)[1]
  input_type = itk.Image[s,d]
  output_type = input_type if outputImageType is None else itk.Image[outputImageType,d]
  thresholder = itk.BinaryThresholdImageFilter[input_type, output_type].New()
  thresholder.SetInput(inputImage)
  thresholder.SetLowerThreshold( lowerThreshold )
  thresholder.SetUpperThreshold( upperThreshold )
  thresholder.SetInsideValue(insideValue)
  thresholder.SetOutsideValue(outsideValue)
  thresholder.Update()
  return thresholder.GetOutput()


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
  s,d = itk.template(inputImage)[1]
  input_type = itk.Image[s,d]
  output_type = input_type if outputImageType is None else itk.Image[outputImageType,d]
  relabel = itk.RelabelComponentImageFilter[input_type, output_type].New()
  relabel.SetInput(inputImage)
  relabel.Update()
  return relabel.GetOutput()


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
  distanceMap = fast_distance_matrix.ManhattanChamferDistance(distanceMapPad, distanceMap.shape)
  distanceMap = distanceMap[1:-1, 1:-1, 1:-1].copy()
  return distanceMap


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
  UCType = itk.ctype('unsigned char')
  FType = itk.ctype('float')
  FloatImageType = itk.Image[FType,3]

  # Read Dicoms Series and turn it into 3D object
  inputCT, Inputmetadata = dicomsTo3D(DicomDir, ShortType)

  print("Preprocessing")
#%%

  showSome(inputCT,50)

  # Preprocessing
  print("Thresholding input image")
  thresholdedInputCT = thresholding(inputCT, lowerThreshold, upperThreshold) # Checked

  showSome(thresholdedInputCT, 50)
  smallScaleSheetnessImage = multiscaleSheetness(multiScaleInput = castImage(thresholdedInputCT, OutputType=FloatImageType),
                                                 scales = [sigmaSmallScale],
                                                 SmoothingImageType = FloatImageType)

  showSome(smallScaleSheetnessImage,50)

  print("Estimating soft-tissue voxels")
  smScale = binaryThresholding(inputImage = smallScaleSheetnessImage,
                               lowerThreshold = -0.05,
                               upperThreshold = 0.05,
                               outputImageType = UCType)
  smScale = ConnectedComponents(inputImage = smScale,
                                outputImageType = itk.ctype('unsigned long'))
  smScale = RelabelComponents(inputImage = smScale,
                              outputImageType = UCType)
  softTissueEstimation = binaryThresholding(inputImage = smScale,
                                            lowerThreshold = 1,
                                            upperThreshold = 1)
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
  autoROI  = binaryThresholding(inputImage = boneDist,
                                lowerThreshold = 0,
                                upperThreshold = 30,
                                outputImageType = UCType)
  print("Unsharp masking")
  InputCT_float = castImage(inputCT, OutputType=FloatImageType)
  # I*G (discrete gauss)
  m_DiffusionFilter = Gaussian(GaussInput = InputCT_float, sigma = 1.0)
  # I - (I*G)
  m_SubstractFilter = substract(InputCT_float, m_DiffusionFilter)
  # k(I-(I*G))
  m_MultiplyFilter = linearTransform(m_SubstractFilter,
                                     scale = 10.,
                                     shift = 0.)
  # I+k*(I-(I*G))
  inputCTUnsharpMasked = add(InputCT_float, m_MultiplyFilter)
  print("Computing multiscale sheetness measure at %d scales" % len(sigmasLargeScale))
  Sheetness = multiscaleSheetness(inputCTUnsharpMasked,
                                  scales = sigmasLargeScale,
                                  SmoothingImageType = FloatImageType,
                                  roi = autoROI)
#%%
  # Pre-Processing Done.




  print(autoROI, Sheetness, softTissueEstimation)






####################
# WORK IN PROGRESS #
####################
#%%
import os
os.listdir("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/")
# Ok: cpp_thresholded = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/thresholdedInputCT.nii", t = 'short')
# Ok: cpp_thresholdedCast = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/thresholdedInputCT_short.nii", t = 'float')
# Ok within float precision: cpp_SmoothingA = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/SmoothinRecursive1.500000.nii", t= 'float')
# Ok within float precision: cpp_smallScaleSheetnessImage = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/smallScaleSheetnessImage.nii", t= 'float')
# Ok within float precision: cpp_soft = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/soft-tissue-est.nii")
# Ok within float precision: cpp_boneEstimation = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/boneEstimation.nii")
# Ok within float precision: cpp_chamferResult = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/chamferResult.nii", t = 'float')
# Ok within previous float precision: cpp_roi = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/roi.nii")
cpp_sheetness = Read3DNifti("/home/PERSONALE/daniele.dallolio3/LOCAL_TOOLS/bone-segmentation/src/proposed-method/src/build/tempfld/sheetnessF.nii", t = 'float')


prova_me = itk.GetArrayFromImage(Sheetness)
prova_cpp = itk.GetArrayFromImage(cpp_sheetness)
showSome(Sheetness,0)
showSome(cpp_sheetness,0)
np.unique(prova_me == prova_cpp)
np.unique(prova_me).max()
np.unique(prova_cpp).max()
np.unique(prova_me).min()
np.unique(prova_cpp).min()


#%%
COST_AMPLIFIER = 1000
# assignIdsToPixels
intensity  = itk.GetArrayFromImage(inputCT)
softTissue = itk.GetArrayFromImage(softTissueEstimation)
sheetness  = itk.GetArrayFromImage(Sheetness)

_pixelIdImage = np.zeros(intensity.shape)
roi = itk.GetArrayFromImage(autoROI)
_pixelIdImage[roi==0] = -1
_totalPixelsInROI = np.sum(roi!=0)
_pixelIdImage[roi!=0] = range(_totalPixelsInROI)


# SheetnessBasedDataCost_compute:
# - BONE, 0
dataCostSource = np.zeros(intensity.shape)
cond = (intensity < -500) | (softTissue == 1)
dataCostSource[cond] = 1 * COST_AMPLIFIER
# - TISSUE, 1
dataCostSink = np.zeros(intensity.shape)
cond = (intensity > 400) & (sheetness > 0)
dataCostSink[cond] = 1 * COST_AMPLIFIER

# SheetnessBasedSmoothCost_compute:
def SheetnessBasedSmoothCost(pixelLeft,
                             pixelRight,
                             shtnLeft,
                             shtnRight):
  alpha = 5.0
  cond = (pixelLeft > -1) & (pixelRight > -1)
  smoothCostFromCenter = np.ones(pixelLeft[cond].shape)
  smoothCostToCenter   = np.ones(pixelLeft[cond].shape)
  dSheet = abs(shtnLeft[cond] - shtnRight[cond])
  # From Center
  cond_b = shtnLeft[cond] >= shtnRight[cond]
  smoothCostFromCenter[cond_b] = np.exp(- 5. * dSheet[cond_b])
  smoothCostFromCenter = smoothCostFromCenter * COST_AMPLIFIER * alpha + 1
  # To Center
  cond_b = np.logical_not(cond_b)
  smoothCostToCenter[cond_b] = np.exp(- 5. * dSheet[cond_b])
  smoothCostToCenter = smoothCostToCenter * COST_AMPLIFIER * alpha + 1
  return pixelLeft[cond], smoothCostFromCenter, smoothCostToCenter

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
_totalNeighbors = np.sum( [len(Xcenters), len(Ycenters), len(Zcenters)] )

# _pixelIdImage[:, :-1, :], _pixelIdImage[:, 1:, :]
# _pixelIdImage[:-1, :, :], _pixelIdImage[1:, :, :]

prova = np.array([[1,2,3,4,5],
                 [6,7,8,9,10],
                 [11,12,13,14,15],
                 [16,17,18,19,20],
                 [21,22,23,24,25]]
                 )



# initializeDataCosts



#%%


###########
# THE END #
###########
