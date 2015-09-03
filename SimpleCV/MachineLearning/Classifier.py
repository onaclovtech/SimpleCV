
from SimpleCV.base import *
from SimpleCV.ImageClass import Image, ImageSet
from SimpleCV.DrawingLayer import *
from SimpleCV.Features import FeatureExtractorBase

class Classifier:

    def _trainPath(self,path,className,subset,disp,verbose):
        count = 0
        files = []
        for ext in IMAGE_FORMATS:
            files.extend(glob.glob( os.path.join(path, ext)))
        if(subset > 0):
            nfiles = min(subset,len(files))
        else:
            nfiles = len(files)
        badFeat = False
        for i in range(nfiles):
            infile = files[i]
            if verbose:
                print "Opening file: " + infile
            img = Image(infile)
            featureVector = []
            for extractor in self.mFeatureExtractors:
                feats = extractor.extract(img)
                if( feats is not None ):
                    featureVector.extend(feats)
                else:
                    badFeat = True

            if(badFeat):
                badFeat = False
                continue

            featureVector.extend([className])
            self.mDataSetRaw.append(featureVector)
            text = 'Training: ' + className
            self._WriteText(disp,img,text,Color.WHITE)
            count = count + 1
            del img
        return count

    def _trainImageSet(self,imageset,className,subset,disp,verbose):
        count = 0
        badFeat = False
        if (subset>0):
            imageset = imageset[0:subset]   
        for img in imageset:
            if verbose:
                print "Opening file: " + img.filename
            featureVector = []
            for extractor in self.mFeatureExtractors:
                feats = extractor.extract(img)
                if( feats is not None ):
                    featureVector.extend(feats)
                else:
                    badFeat = True
                    
            if(badFeat):
                badFeat = False
                continue
            
            featureVector.extend([className])
            self.mDataSetRaw.append(featureVector)
            text = 'Training: ' + className
            self._WriteText(disp,img,text,Color.WHITE)
            count = count + 1
            del img
        return count
    def save(self, fname):
        """
        Save the classifier to file
        """
        output = open(fname, 'wb')
        pickle.dump(self,output,2) # use two otherwise it w
        output.close()

    def load(cls, fname):
        """
        Load the classifier from file
        """
        return pickle.load(file(fname, 'rb'))
    load = classmethod(load)

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.mDataSetOrange = None
        del mydict['mDataSetOrange']
        self.mOrangeDomain = None
        del mydict['mOrangeDomain']
        return mydict
