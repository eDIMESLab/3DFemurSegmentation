import itk
import numpy as np
import ctypes
import matplotlib.pylab as plb
from scipy.ndimage.morphology import distance_transform_cdt


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
              threshold = 1e-2):
  eigenvalues_matrix = np.abs(itk.GetArrayFromImage(RsInputImg))
  eigenvalues_matrix[:,:,:,:3] = np.sort(eigenvalues_matrix[:,:,:,:3])
  det_image = np.sum(eigenvalues_matrix, axis=-1)
  mean_norm = 1. / np.mean(det_image)
  Rnoise = det_image*mean_norm
  RsImage = np.empty(eigenvalues_matrix.shape[:-1] + (4,), dtype=float)
  al3_null = eigenvalues_matrix[:,:,:,2] > threshold
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
                            alpha = 0.5,
                            beta = 0.5,
                            gamma = 0.5):
  RsImg, EigsImg, NoNullEigs = computeRs(RsInputImg = SheetMeasInput)
  SheetnessImage = np.empty(EigsImg.shape[:-1], dtype=float)
  SheetnessImage[NoNullEigs] = - np.sign( EigsImg[NoNullEigs,2] )
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




def computeEigenvalues(HessianObj,
                       EigType,
                       D = 3):
  HessianImageType = type(HessianObj)
  EigenValueArrayType = itk.FixedArray[EigType, D]
  EigenValueImageType = itk.Image[EigenValueArrayType, D]
  EigenAnalysisFilterType = itk.SymmetricEigenAnalysisImageFilter[HessianImageType]

  m_EigenAnalysisFilter = EigenAnalysisFilterType.New()
  m_EigenAnalysisFilter.SetDimension(D)
  m_EigenAnalysisFilter.SetInput(HessianObj)
  m_EigenAnalysisFilter.Update()

  return m_EigenAnalysisFilter.GetOutput()


def computeHessian(HessInput,
                   sigma,
                   HRGImageType):
  HessianFilterType = itk.HessianRecursiveGaussianImageFilter[HRGImageType]
  HessianObj = HessianFilterType.New()
  HessianObj.SetSigma(sigma)
  HessianObj.SetInput(HessInput)
  HessianObj.Update()
  return HessianObj.GetOutput()


def singlescaleSheetness(singleScaleInput,
                         scale,
                         HessImageType,
                         roi = None,
                         alpha = 0.5,
                         beta = 0.5,
                         gamma = 0.5):
  print("Computing single-scale sheetness, sigma=%4.2f"% scale)
  HessImg = computeHessian(HessInput = singleScaleInput,
                           sigma = scale,
                           HRGImageType = HessImageType)
  EigenImg = computeEigenvalues(HessianObj = HessImg,
                                EigType = itk.ctype('float'),
                                D = 3)
  SheetnessImg = computeSheetnessMeasure(SheetMeasInput = EigenImg,
                                         alpha = alpha,
                                         beta = beta,
                                         gamma = gamma)

  if not roi is None:
    SheetnessImg[roi==0] = 0.
  return SheetnessImg





def multiscaleSheetness(multiScaleInput,
                        scales,
                        HessImageType,
                        roi = None,
                        alpha = 0.5,
                        beta = 0.5,
                        gamma = 0.5):
  if not roi is None:
    roi = itk.GetArrayFromImage(roi)
  multiscaleSheetness = singlescaleSheetness(singleScaleInput = multiScaleInput,
                                             scale = scales[0],
                                             HessImageType = HessImageType,
                                             roi = roi,
                                             alpha = alpha,
                                             beta = beta,
                                             gamma = gamma)

  if len(scales) > 1:
    for scale in scales[1:]:
      singleScaleSheetness  = singlescaleSheetness(multiScaleInput,
                                                   scale = scale,
                                                   HessImageType = HessImageType,
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


#%%
import matplotlib.pylab as plb
def showSome(imgObj, idx = 0):
  prova = itk.GetArrayFromImage(imgObj)
  plb.imshow(prova[idx,:,:])
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

#%%

  # Preprocessing
  print("Thresholding input image")
  thresholdedInputCT = thresholding(inputCT, lowerThreshold, upperThreshold)
  smallScaleSheetnessImage = multiscaleSheetness(castImage(thresholdedInputCT, OutputType=FloatImageType),
                                                 scales = [sigmaSmallScale],
                                                 HessImageType = FloatImageType)
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


  print("Estimating bone voxels")
  boneEstimation = itk.GetArrayFromImage(inputCT)
  smScale = itk.GetArrayFromImage(smallScaleSheetnessImage)
  boneCondition = (boneEstimation > 400) | (boneEstimation > 250) & (smScale > 0.6)
  boneEstimation[boneCondition] = 1
  boneEstimation[np.logical_not(boneCondition)] = 0
  print("Computing ROI from bone estimation using Chamfer Distance")
  boneDist = distance_transform_cdt(boneEstimation,
                                    metric='manhattan',
                                    return_distances=True).astype(np.float32)
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
                                  HessImageType = FloatImageType,
                                  roi = autoROI)
#%%
  # Pre-Processing Done.




  print(autoROI, Sheetness, softTissueEstimation)







###########
# THE END #
###########
