import itk
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import ctypes
#import matplotlib.pylab as plb
from mpl_toolkits import mplot3d
import os
import sys
#%%
def MaskOriginal(OriginalImg, FinalSegm):
  OriginalImg_np = itk.GetArrayFromImage(OriginalImg)
  FinalSegm_np = itk.GetArrayFromImage(FinalSegm)
  OriginalImg_np[FinalSegm_np<1] = 0.
  OriginalImg_masked = itk.GetImageFromArray(OriginalImg_np)
  OriginalImg_masked.SetOrigin(OriginalImg.GetOrigin())
  OriginalImg_masked.SetSpacing(OriginalImg.GetSpacing())
  OriginalImg_masked.SetDirection(OriginalImg.GetDirection())
  return OriginalImg_masked

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
  seriesGenerator.SetDirectory(dirname)

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

def GetBoundaries(img, ImageType, back_value=0, fore_value=1):
  binaryContourImageFilterType = itk.BinaryContourImageFilter[ImageType,ImageType]
  binaryContourFilter = binaryContourImageFilterType.New()
  binaryContourFilter.SetInput(img)
  binaryContourFilter.SetBackgroundValue(back_value)
  binaryContourFilter.SetForegroundValue(fore_value)
  binaryContourFilter.Update()
  return itk.GetArrayFromImage(binaryContourFilter.GetOutput())

# Hausdorff Distance
def HausdorffDistance(Input1, Input2, ImageType, spacing=True, average=False):
  HausdorffDistanceFilterType = itk.HausdorffDistanceImageFilter[ImageType,ImageType]
  HausdorffDistanceFilter = HausdorffDistanceFilterType.New(Input1,Input2)
  HausdorffDistanceFilter.SetUseImageSpacing(spacing)
  HausdorffDistanceFilter.Update()
  if average:
    return HausdorffDistanceFilter.GetAverageHausdorffDistance()
  else:
    return HausdorffDistanceFilter.GetHausdorffDistance()

def castImage(imgObj, OutputType):
  s,d = itk.template(imgObj)[1]
  input_type = itk.Image[s,d]
  output_type = OutputType
  castObj = itk.CastImageFilter[input_type, output_type].New()
  castObj.SetInput(imgObj)
  castObj.Update()
  return castObj.GetOutput()

#%%
# Uncomment when needed:
import matplotlib.pylab as plb
def showSome(imgObj, idx = 0):
  prova = itk.GetArrayFromImage(imgObj)
  plb.imshow(prova[idx,:,:])


#%%

DicomDir = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_CTData"
GCSegmDir = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_Masked"
ManualSegm_L_file = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_L-label.nrrd"
ManualSegm_R_file = "/home/PERSONALE/daniele.dallolio3/femur_segmentation/D0012_R-label.nrrd"

ShortType = itk.ctype('short')
ShortImageType = itk.Image[ShortType,3]
# Read Unsupervised and Supervised segmentations
inputCT, Inputmetadata = dicomsTo3D(DicomDir, ShortType)


GCSegm, GCSegm_metadata = dicomsTo3D(GCSegmDir, ShortType)
ManualSegm_L, ManualSegm_metadata_L = readNRRD(ManualSegm_L_file, ShortType)
ManualSegm_R, ManualSegm_metadata_R = readNRRD(ManualSegm_R_file, ShortType)
ManualSegm = itk.GetImageFromArray(itk.GetArrayFromImage(ManualSegm_L) | itk.GetArrayFromImage(ManualSegm_R))


#%%
# Define the estimated femurs
GCSegm_arr = itk.GetArrayFromImage(GCSegm)
all_labels = {i: 0 for i in np.unique(GCSegm_arr) if i>0}
for z in range(0, GCSegm_arr.shape[0]):
  for i in np.unique(GCSegm_arr[z,:,:]):
    if i>0:
      all_labels[i] +=1
two_femur = list({k: v for k, v in sorted(all_labels.items(), key=lambda item: item[1], reverse=True)}.keys())[0:2]
GCSegm_arr[(GCSegm_arr != two_femur[0]) & (GCSegm_arr != two_femur[1])] = 0
GCSegm_arr[GCSegm_arr>0] = 1
GCSegm = itk.GetImageFromArray(GCSegm_arr)

#%%
# Compute femurs' boundaries
GCcontours = itk.GetImageFromArray(GetBoundaries(GCSegm, ShortImageType, 0, 1).astype(np.float32))
GCcontours.SetOrigin(inputCT.GetOrigin())
GCcontours.SetSpacing(inputCT.GetSpacing())
GCcontours.SetDirection(inputCT.GetDirection())
Manualcontours = itk.GetImageFromArray(GetBoundaries(ManualSegm, ShortImageType, 0, 1).astype(np.float32))
Manualcontours.SetOrigin(inputCT.GetOrigin())
Manualcontours.SetSpacing(inputCT.GetSpacing())
Manualcontours.SetDirection(inputCT.GetDirection())

#%%
showSome(GCSegm, 0)
showSome(ManualSegm, 0)
showSome(GCcontours, 400)
showSome(Manualcontours, 400)

#%%
FType = itk.ctype('float')
FloatImageType = itk.Image[FType,3] # 3 is the number of dimension!

HausdorffDistance(Input1=GCcontours, Input2=Manualcontours, ImageType=FloatImageType)

prova1 = itk.GetArrayFromImage(GCcontours)[400,:,:]
prova2 = itk.GetArrayFromImage(Manualcontours)[400,:,:]
prova1_itk = itk.GetImageFromArray(prova1.astype(np.float32))
prova2_itk = itk.GetImageFromArray(prova2.astype(np.float32))
prova1_itk.SetOrigin( list(inputCT.GetOrigin())[0:2] )
prova1_itk.SetSpacing( list(inputCT.GetSpacing())[0:2] )
prova2_itk.SetOrigin( list(inputCT.GetOrigin())[0:2] )
prova2_itk.SetSpacing( list(inputCT.GetSpacing())[0:2] )

FloatImageType_2d = itk.Image[FType,2]
HausdorffDistance(Input1=prova1_itk, Input2=prova2_itk, ImageType=FloatImageType_2d)
#%%
# Closest Euclidean distance from any non-zero pixel
def SignedMaurerDistanceMap(img, ImageType, spacing = True):
  SignedMaurerDistanceMapImageFilterType = itk.SignedMaurerDistanceMapImageFilter[ImageType, ImageType]
  SignedMaurerDistanceMapImageFilter = SignedMaurerDistanceMapImageFilterType.New()
  SignedMaurerDistanceMapImageFilter.SetUseImageSpacing(spacing)
  SignedMaurerDistanceMapImageFilter.SetInput(img)
  SignedMaurerDistanceMapImageFilter.Update()
  return SignedMaurerDistanceMapImageFilter.GetOutput()

def LDMap(Input1, Input2, ImageType, spacing = True):
  distance_1 = itk.GetArrayFromImage(SignedMaurerDistanceMap(Input1, ImageType, spacing))
  distance_2 = itk.GetArrayFromImage(SignedMaurerDistanceMap(Input2, ImageType, spacing))
  A1 = (distance_1>1e-5).astype(np.float32)
  B1 = (distance_2>1e-5).astype(np.float32)
  LDMap_out = np.abs(A1 - B1) * np.maximum( distance_1, distance_2 )
  return LDMap_out


# Hausdorff local maps
LDMap_final = LDMap(Input1=GCcontours, Input2=Manualcontours, ImageType=FloatImageType)
plb.imshow(LDMap_final[310,:,:])

#%%
# Max HD per slice
hd_slices = []
for z in range(LDMap_final.shape[0]):
  hd_slices += [np.max(LDMap_final[z,:,:])]

#%%
# Hausdorff map considering binary images (not only contours)
LDMap_finalNoBounds = LDMap(Input1=castImage(GCSegm, FloatImageType), Input2=castImage(ManualSegm, FloatImageType), ImageType=FloatImageType)
plb.imshow(np.sqrt(LDMap_finalNoBounds[399,:,:]))


#%%
