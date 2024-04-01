import numpy as np
import xml.etree.ElementTree as ET


class ROI:
    def __init__(self, width, height, stride):
        self.width = width
        self.height = height
        self.stride = stride
        self.X = 0
        self.Y = 0
        self.xbin = 1
        self.ybin = 1


class MetaContainer:
    def __init__(self, metaType, stride, *, metaEvent: str = '', metaResolution: np.int64 = 0):
        self.metaType = metaType
        self.stride = stride
        self.metaEvent = metaEvent
        self.metaResolution = metaResolution


class dataContainer:
    def __init__(self, data, **kwargs):
        self.data = data
        self.__dict__.update(kwargs)


def read_spe(filePath):
    dataTypes = {'MonochromeUnsigned16': np.uint16, 'MonochromeUnsigned32': np.uint32,
                 'MonochromeFloating32': np.float32}

    with open(filePath) as f:
        f.seek(678)
        xmlLoc = np.fromfile(f, dtype=np.uint64, count=1)[0]
        f.seek(1992)
        speVer = np.fromfile(f, dtype=np.float32, count=1)[0]

    if speVer == 3:
        with open(filePath, encoding="utf8") as f:
            f.seek(xmlLoc)
            xmlFooter = f.read()
            xmlRoot = ET.fromstring(xmlFooter)
            regionList = list()
            metaList = list()
            dataList = list()
            # print(xmlRoot[0][0].attrib)
            calFlag = False
            for child in xmlRoot:
                if 'DataFormat'.casefold() in child.tag.casefold():
                    for child1 in child:
                        if 'DataBlock'.casefold() in child1.tag.casefold():
                            readoutStride = np.int64(child1.get('stride'))
                            numFrames = np.int64(child1.get('count'))
                            pixFormat = child1.get('pixelFormat')
                            for child2 in child1:
                                if 'DataBlock'.casefold() in child1.tag.casefold():
                                    regStride = np.int64(child2.get('stride'))
                                    regWidth = np.int64(child2.get('width'))
                                    regHeight = np.int64(child2.get('height'))
                                    regionList.append(ROI(regWidth, regHeight, regStride))
                if 'MetaFormat'.casefold() in child.tag.casefold():
                    for child1 in child:
                        if 'MetaBlock'.casefold() in child1.tag.casefold():
                            for child2 in child1:
                                metaType = child2.tag.rsplit('}', maxsplit=1)[1]
                                metaEvent = child2.get('event')
                                metaStride = np.int64(np.int64(child2.get('bitDepth')) / 8)
                                metaResolution = child2.get('resolution')
                                if metaEvent != None and metaResolution != None:
                                    metaList.append(MetaContainer(metaType, metaStride, metaEvent=metaEvent,
                                                                  metaResolution=np.int64(metaResolution)))
                                else:
                                    metaList.append(MetaContainer(metaType, metaStride))
                if 'Calibrations'.casefold() in child.tag.casefold():
                    for child1 in child:
                        if 'WavelengthMapping'.casefold() in child1.tag.casefold():
                            for child2 in child1:
                                if 'WavelengthError'.casefold() in child2.tag.casefold():
                                    wavelengths = np.array([])
                                    wlText = child2.text.rsplit()
                                    for elem in wlText:
                                        wavelengths = np.append(wavelengths, np.fromstring(elem, sep=',')[0])
                                else:
                                    wavelengths = np.fromstring(child2.text, sep=',')
                            calFlag = True

            regionOffset = 0
            # read entire datablock
            f.seek(0)
            bpp = np.dtype(dataTypes[pixFormat]).itemsize
            numPixels = np.int32((xmlLoc - 4100) / bpp)
            totalBlock = np.fromfile(f, dtype=dataTypes[pixFormat], count=numPixels, offset=4100)
            for i in range(0, len(regionList)):
                offLen = list()
                if i > 0:
                    regionOffset += (regionList[i - 1].stride) / bpp
                for j in range(0, numFrames):
                    offLen.append((np.int32(regionOffset + (j * readoutStride / bpp)),
                                   regionList[i].width * regionList[i].height))
                regionData = np.concatenate([totalBlock[offset:offset + length] for offset, length in offLen])
                dataList.append(
                    np.reshape(regionData, (numFrames, regionList[i].height, regionList[i].width), order='C'))

            if calFlag == False:
                totalData = dataContainer(dataList, xmlFooter=xmlFooter)
            else:
                totalData = dataContainer(dataList, xmlFooter=xmlFooter, wavelengths=wavelengths)
            return totalData


    elif speVer < 3:
        header = {}
        adc_byte = {0: np.float32, 1: np.int32, 2: np.int16, 3: np.uint16, 5: np.float64, 6: np.uint8, 8: np.uint32}
        dataTypes2 = {0: np.float32, 1: np.int32, 2: np.int16, 3: np.uint16, 5: np.float64, 6: np.uint8, 8: np.uint32}
        with open(filePath, encoding="utf8") as f:
            # read metadata
            f.seek(0)
            header['ControlerVer'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 0
            header['LogicOutput'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 2
            header['AmpHiCapLowNois'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 4
            header['xDimCCD'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 6
            header['mode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 8
            header['ExpTime'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 10
            header['ChipXdim'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 14
            header['ChipYdim'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 16
            header['Ydim'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 18
            header['date'] = np.fromfile(f, dtype=np.byte, count=10).tostring().decode().replace('\x00', '')  # 20
            header['VirtualChipFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 30
            header['Spare_1'] = np.fromfile(f, dtype=np.byte, count=2).tostring().decode().replace('\x00', '')  # 32
            header['noscan'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 34
            header['DetTemperature'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 36
            header['DetType'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 40
            header['xdim'] = np.int64(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 42
            header['stdiod'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 44
            header['DelayTime'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 46
            header['ShutterControl'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 50
            header['AbsorbLive'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 52
            header['AbsorbMode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 54
            header['CanDoFirtualChipFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 56
            header['ThresholdMinLive'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 58
            header['ThresholdMinVal'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 60
            header['ThresholdMaxLive'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 64
            header['ThresholdMaxVal'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 66
            header['SpecAutoSpectroMode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 70
            header['SpecCenterWlNm'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 72
            header['SpecGlueFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 76
            header['SpecGlueStasrtWlNm'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 78
            header['SpecGlueEndWlNm'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 82
            header['SpecGlueMinOvrlpNm'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 86
            header['SpecGlueFinalResNm'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 90
            header['PulserType'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 94
            header['CustomChipFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 96
            header['XPrePixels'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 98
            header['XPostPixels'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 100
            header['YPrePixels'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 102
            header['YPostPixels'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 104
            header['asynen'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 106
            header['datatype'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 108
            header['PulserMode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 110
            header['PulserOnChipAccums'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 112
            header['PulserRepeatExp'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 114
            header['PulseRepWidth'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 118
            header['PulseRepDelay'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 122
            header['PulseSeqStartWidth'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 126
            header['PulseSeqEndWidth'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 130
            header['PulseSeqStartDelay'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 134
            header['PulseSeqEndDelay'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 138
            header['PulseSeqIncMode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 142
            header['PImaxUsed'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 144
            header['PImaxMode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 146
            header['PImaxGain'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 148
            header['BackGrndApplied'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 150
            header['PImax2nsBrdUsed'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 152
            header['minblk'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 154
            header['numminblk'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 156
            header['SpecMirrorLocation'] = np.int16(np.fromfile(f, dtype=np.uint16, count=2)[0])  # 158
            header['SpecSlitLocation'] = np.int16(np.fromfile(f, dtype=np.uint16, count=4)[0])  # 162
            header['CustomTimingFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 170
            header['ExperimentTimeLocal'] = np.fromfile(f, dtype=np.byte, count=7).tostring().decode()  # 172
            header['ExperimentTimeUTC'] = np.fromfile(f, dtype=np.byte, count=7).tostring().decode()  # 179
            header['ExposUnits'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 186
            header['ADCoffset'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 188
            header['ADCrate'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 190
            header['ADCtype'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 192
            header['ACDresolution'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 194
            header['ACDbit'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 196
            header['gain'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 198
            header['Comments'] = np.fromfile(f, dtype=np.byte, count=400).tostring().decode().replace('\x00',
                                                                                                      '')  # 200
            header['geometric'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 600
            header['xlabel'] = np.fromfile(f, dtype=np.byte, count=16).tostring().decode().replace('\x00', '')  # 602
            header['cleans'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 618
            header['NumSkpPerCln'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 620
            header['SpecMirrorPos'] = np.array(np.fromfile(f, dtype=np.uint8, count=2)[:])  # 624
            header['SpecSlitPos'] = np.array(np.fromfile(f, dtype=np.uint32, count=4)[:])  # 626
            header['AutoCleansActive'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 642
            header['UseContCleansInst'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 644
            header['AbsorbStripNum'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 646
            header['SpecSlitPosUnits'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 648
            header['SpecGrooves'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 650
            header['srccmp'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 654
            header['ydim'] = np.int64(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 656
            header['scramble'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 658
            header['ContinuosCleansFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 660
            header['ExternalTriggerFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 662
            header['lnoscan'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 664
            header['lavgexp'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 668
            header['ReadoutTime'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 672
            header['TriggerModeFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 676
            header['Spare_2'] = np.fromfile(f, dtype=np.byte, count=10).tostring().decode().replace('\x00', '')  # 678
            header['sw_version'] = np.fromfile(f, dtype=np.byte, count=16).tostring().decode().replace('\x00',
                                                                                                       '')  # 688
            header['type'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 704
            header['flagFieldApplied'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 706
            header['Spare_3'] = np.fromfile(f, dtype=np.byte, count=16).tostring().decode().replace('\x00', '')  # 708
            header['kin_trig_mode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 724
            header['dlabel'] = np.fromfile(f, dtype=np.byte, count=16).tostring().decode().replace('\x00', '')  # 726
            header['Spare_4'] = np.fromfile(f, dtype=np.byte, count=436).tostring().decode().replace('\x00', '')  # 742
            header['PulserFileName'] = np.fromfile(f, dtype=np.byte, count=120).tostring().decode().replace('\x00',
                                                                                                            '')  # 1178
            header['AbsorbFileName'] = np.fromfile(f, dtype=np.byte, count=120).tostring().decode().replace('\x00',
                                                                                                            '')  # 1298
            header['NumExpRepeats'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 1418
            header['NumExpAccums'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 1422
            header['YT_Flag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1426
            header['clkspd_us'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 1428
            header['HWaccumFlag'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1432
            header['StoreSync'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1434
            header['BlemishAppied'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1436
            header['CosmicApplied'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1438
            header['CosmicType'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1440
            header['CosmicThreshold'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 1442
            header['NumFrames'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 1446
            header['MaxIntensity'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 1450
            header['MinIntensity'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 1454
            header['ylabel'] = np.fromfile(f, dtype=np.byte, count=16).tostring().decode().replace('\x00', '')  # 1458
            header['ShutterType'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1474
            header['shutterComp'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 1476
            header['readoutMode'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1480
            header['WindowSize'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1482
            header['clkspd'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1484
            header['interface_type'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1486
            header['NumROIsInExperiment'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1488
            header['Spare_5'] = np.fromfile(f, dtype=np.byte, count=16).tostring().decode().replace('\x00', '')  # 1490
            header['controllerNum'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1506
            header['SWmade'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1508
            header['NumROI'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # 1510
            header['ROI_1'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1512
            header['ROI_2'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1524
            header['ROI_3'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1536
            header['ROI_4'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1548
            header['ROI_5'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1560
            header['ROI_6'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1572
            header['ROI_7'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1584
            header['ROI_8'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1596
            header['ROI_9'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1608
            header['ROI_10'] = np.array(np.fromfile(f, dtype=np.uint16, count=6)[:])  # 1620
            header['FlatField'] = np.fromfile(f, dtype=np.byte, count=120).tostring().decode().replace('\x00',
                                                                                                       '')  # 1632
            header['background'] = np.fromfile(f, dtype=np.byte, count=120).tostring().decode().replace('\x00',
                                                                                                        '')  # 1752
            header['blemish'] = np.fromfile(f, dtype=np.byte, count=120).tostring().decode().replace('\x00', '')  # 1872
            header['file_header_ver'] = np.float32(np.fromfile(f, dtype=np.float32, count=1)[0])  # 1992
            header['YT_Info'] = np.fromfile(f, dtype=np.byte, count=1000).tostring().decode().replace('\x00',
                                                                                                      '')  # 1996
            header['WinView_id'] = np.int32(np.fromfile(f, dtype=np.uint32, count=1)[0])  # 2996
            header['Xoffset'] = np.float64(np.fromfile(f, dtype=np.float64, count=1)[0])  # 3000
            header['Xfactor'] = np.float64(np.fromfile(f, dtype=np.float64, count=1)[0])  # 3008
            header['Xcurrent_unit'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3016
            header['Xreserved1'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3017
            header['Xstring'] = np.fromfile(f, dtype=np.byte, count=40).tostring().decode().replace('\x00', '')  # 3018
            header['Xreserved2'] = np.fromfile(f, dtype=np.byte, count=40).tostring().decode().replace('\x00',
                                                                                                       '')  # 3058
            header['Xcalib_valid'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3098
            header['Xinput_unit'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3099
            header['Xpolynom_unit'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3100
            header['Xpolymon_order'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3101
            header['Xcalib_count'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3102
            header['Xpixel_position'] = np.array(np.fromfile(f, dtype=np.float64, count=10)[:])  # 3103
            header['Xcalib_value'] = np.array(np.fromfile(f, dtype=np.float64, count=10)[:])  # 3183
            header['Xpolynom_coeff'] = np.array(np.fromfile(f, dtype=np.float64, count=6)[:])  # 3263
            header['Xlaser_position'] = np.float64(np.fromfile(f, dtype=np.float64, count=1)[0])  # 3311
            header['Xreserved3'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3319
            header['Xnew_calib_flag'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3320
            header['Xcalib_label'] = np.fromfile(f, dtype=np.byte, count=81).tostring().decode().replace('\x00',
                                                                                                         '')  # 3321
            header['Xexpansion'] = np.fromfile(f, dtype=np.byte, count=87).tostring().decode().replace('\x00',
                                                                                                       '')  # 3402
            header['Yoffset'] = np.float64(np.fromfile(f, dtype=np.float64, count=1)[0])  # 3489
            header['Yfactor'] = np.float64(np.fromfile(f, dtype=np.float64, count=1)[0])  # 3497
            header['Ycurrent_unit'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3505
            header['Yreserved1'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3506
            header['Ystring'] = np.fromfile(f, dtype=np.byte, count=40).tostring().decode().replace('\x00', '')  # 3507
            header['Yreserved2'] = np.fromfile(f, dtype=np.byte, count=40).tostring().decode().replace('\x00',
                                                                                                       '')  # 3547
            header['Ycalib_valid'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3587
            header['Yinput_unit'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3588
            header['Ypolynom_unit'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3589
            header['Ypolymon_order'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3590
            header['Ycalib_count'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3591
            header['Ypixel_position'] = np.array(np.fromfile(f, dtype=np.float64, count=10)[:])  # 3692
            header['Ycalib_value'] = np.array(np.fromfile(f, dtype=np.float64, count=10)[:])  # 3672
            header['Ypolynom_coeff'] = np.array(np.fromfile(f, dtype=np.float64, count=6)[:])  # 3752
            header['Ylaser_position'] = np.float64(np.fromfile(f, dtype=np.float64, count=1)[0])  # 3800
            header['Yreserved3'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3808
            header['Ynew_calib_flag'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # 3809
            header['Ycalib_label'] = np.fromfile(f, dtype=np.byte, count=81).tostring().decode().replace('\x00',
                                                                                                         '')  # 3810
            header['Yexpansion'] = np.fromfile(f, dtype=np.byte, count=87).tostring().decode().replace('\x00',
                                                                                                       '')  # 3891
            header['Istring'] = np.fromfile(f, dtype=np.byte, count=40).tostring().decode().replace('\x00',
                                                                                                    '')  # char(fread(f,40,'char'))';%3978
            header['Spare_6'] = np.fromfile(f, dtype=np.byte, count=25).tostring().decode().replace('\x00',
                                                                                                    '')  # char(fread(f,25,'char'))';%4018
            header['SpecType'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # fread(f,1,'char');%4043
            header['SpecModel'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # fread(f,1,'char');%4044
            header['PulseBurstUsed'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # fread(f,1,'char');%4045
            header['PulseBurstCount'] = np.int32(
                np.fromfile(f, dtype=np.uint32, count=1)[0])  # =fread(f,1,'long');%4046
            header['PulseBurstPeriod'] = np.float64(
                np.fromfile(f, dtype=np.float64, count=1)[0])  # fread(f,1,'double');%4050
            header['PulseBracketUsed'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # fread(f,1,'char');%4058
            header['PulseBracketType'] = np.int8(np.fromfile(f, dtype=np.uint8, count=1)[0])  # fread(f,1,'char');%4059
            header['PulseTimeConstFast'] = np.float64(
                np.fromfile(f, dtype=np.float64, count=1)[0])  # fread(f,1,'double');%4060
            header['PulseAmplitudeFast'] = np.float64(
                np.fromfile(f, dtype=np.float64, count=1)[0])  # fread(f,1,'double');%4068
            header['PulseTimeConstSlow'] = np.float64(
                np.fromfile(f, dtype=np.float64, count=1)[0])  # fread(f,1,'double');%4076
            header['PulseAmplitudeSlow'] = np.float64(
                np.fromfile(f, dtype=np.float64, count=1)[0])  # fread(f,1,'double');%4084
            header['AnalogGain'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # fread(f,1,'short');%4092
            header['AvGainUsed'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # fread(f,1,'short');%4094
            header['AvGain'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # fread(f,1,'short');%4096
            header['lastvalue'] = np.int16(np.fromfile(f, dtype=np.uint16, count=1)[0])  # fread(f,1,'short');%4098

            # # read raw data

            num_Pixels = header['xdim'] * header['ydim'] * header['NumFrames']
            totalBlock = np.fromfile(f, dtype=adc_byte[header['datatype']], count=num_Pixels)
            offLen = list()
            for j in range(0, header['NumFrames']):
                offLen.append((np.int32((j * header['xdim'] * header['ydim'])), header['xdim'] * header['ydim']))
            regionData = np.concatenate([totalBlock[offset:offset + length] for offset, length in offLen])
            data = np.reshape(regionData, (header['NumFrames'], header['ydim'], header['xdim']), order='C')
            header['data'] = np.transpose(data, (2, 0, 1))

            return header