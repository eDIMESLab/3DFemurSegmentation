import itk
import numpy as np
import ctypes
import matplotlib.pylab as plb



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
  tmp = 3. / det_image
  RsImage[al3_null, 2] = eigs[:,0]*tmp # Rblob
  RsImage[al3_null, 3] = Rnoise # Rnoise
  return RsImage, eigenvalues_matrix


def computeSheetnessMeasure(SheetMeasInput,
                            alpha = 0.5,
                            beta = 0.5,
                            gamma = 0.5):
  RsImg, EigsImg = computeRs(SheetMeasInput)
  SheetnessImage = np.empty(eigenvalues_matrix.shape[:-1] + (1,), dtype=float)
  SheetnessImage = - np.sign( EigsImg[:,:,:,2] )
  SheetnessImage *= np.exp(-RsImg[:,:,:,1])
  tmp = 1. / (alpha*alpha)
  SheetnessImage *= np.exp(-RsImg[:,:,:,1] * RsImg[:,:,:,1] * tmp)
  tmp = 1. / (beta*beta)
  SheetnessImage *= np.exp(-RsImg[:,:,:,0] * RsImg[:,:,:,0] * tmp)
  tmp = 1. / (gamma*gamma)
  SheetnessImage *= np.exp(-RsImg[:,:,:,2] * RsImg[:,:,:,2] * tmp)
  # SheetnessImage *= EigsImg[:,:,:,2] ScaleObjectnessMeasureOff
  SheetnessImage *= ( 1 - np.exp(-RsImg[:,:,:,3] * RsImg[:,:,:,3] * 4) )
  SheetnessImage = itk.GetImageFromArray(SheetnessImage)
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
  return SheetnessImg






def multiscaleSheetness(multiScaleInput,
                        scales,
                        HessImageType,
                        alpha = 0.5,
                        beta = 0.5,
                        gamma = 0.5):
  multiscaleSheetness = singlescaleSheetness(multiScaleInput,
                                             scales[0],
                                             HessImageType,
                                             alpha = alpha,
                                             beta = beta,
                                             gamma = gamma)
  if len(scales) > 1:
    for scale in scales[1:]:
      singlescaleSheetness(multiScaleInput,
                           scale,
                           HessImageType,
                           alpha = alpha,
                           beta = beta,
                           gamma = gamma)
  else:
    return




def binaryThresholding(inputImage,
                       lowerThreshold,
                       upperThreshold,
                       insideValue,
                       outsideValue,
                       ImageType):
  itk.BinaryThresholdImageFilter[].New()
  thresholder.SetInput(inputImage)
  thresholder.SetLowerThreshold( lowerThreshold )
  thresholder.SetUpperThreshold( upperThreshold )
  thresholder.SetInsideValue(insideValue)
  thresholder.SetOutsideValue(outsideValue)
  thresholder.Update()
  return thresholder.GetOutput()


if __name__ == "__main__":
  #
  # Add parser
  #
  #

  # Parameters
  sigmaSmallScale = 1.5
  lowerThreshold = 25
  upperThreshold = 600

  DicomDir = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/CT_femore_Viceconti/CT_dataset_FemoreEsempio/"
  # DicomDir = "/home/daniele/OneDrive/Lavori/CT_femore_Viceconti/CT_dataset_FemoreEsempio/"

  # Useful shortcut
  ShortType = itk.ctype('short')
  FloatImageType = itk.Image[itk.ctype('float'),3]

  # Read Dicoms Series and turn it into 3D object
  inputCT, Inputmetadata = dicomsTo3D(DicomDir, ShortType)

  # Preprocessing
  thresholdedInputCT = thresholding(inputCT, lowerThreshold, upperThreshold)
  smallScaleSheetnessImage = multiscaleSheetness(castImage(thresholdedInputCT, OutputType=FloatImageType),
                                                 Scales,
                                                 FloatImageType)



###########
# THE END #
###########
