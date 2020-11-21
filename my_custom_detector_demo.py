from MyDetector import TF2Detector

from Myutils import detectvideo

class TF2detectorargs:
    modelbasefolder = './my_trained_model/saved_model/'
    labelmappath = './labels.txt'
    threshold = 0.4

def testTF2Detector(detectorargs):
    mydetector = TF2Detector.MyTF2Detector(detectorargs)

    outputvideo = 'output-opencv-gstreamer.mp4'
    inputvideo = 'inference.MOV'
    detectvideo.detectvideo_tovideo(inputvideo, mydetector, outputvideo)

if __name__ == "__main__":
    testTF2Detector(TF2detectorargs)