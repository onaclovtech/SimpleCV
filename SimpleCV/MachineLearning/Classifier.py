
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
        
    def _testPath(self,path,className,dataset,subset,disp,verbose):
        count = 0
        correct = 0
        badFeat = False
        files = []
        for ext in IMAGE_FORMATS:
            files.extend(glob.glob( os.path.join(path, ext)))
        if(subset > 0):
            nfiles = min(subset,len(files))
        else:
            nfiles = len(files)
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
            if( badFeat ):
                del img
                badFeat = False
                continue
            featureVector.extend([className])
            dataset.append(featureVector)
            test = orange.ExampleTable(self.mOrangeDomain,[featureVector])
            c = self.mClassifier(test[0])
            testClass = test[0].getclass()
            if(testClass==c):
                text =  "Classified as " + str(c)
                self._WriteText(disp,img,text, Color.GREEN)
                correct = correct + 1
            else:
                text =  "Mislassified as " + str(c)
                self._WriteText(disp,img,text, Color.RED)
            count = count + 1
            del img

        return([dataset,count,correct])
        
    def setFeatureExtractors(self, extractors):
        """
        Add a list of feature extractors to the classifier. These feature extractors
        must match the ones used to train the classifier. If the classifier is already
        trained then this method will require that you retrain the data.
        """
        self.mFeatureExtractors = extractors
        return None
        
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
    def __setstate__(self, mydict):
        self.__dict__ = mydict
        colNames = []
        for extractor in self.mFeatureExtractors:
            colNames.extend(extractor.getFieldNames())
        self.mOrangeDomain = orange.Domain(map(orange.FloatVariable,colNames),orange.EnumVariable("type",values=self.mClassNames))
        self.mDataSetOrange = orange.ExampleTable(self.mOrangeDomain,self.mDataSetRaw)


    def _WriteText(self, disp, img, txt,color):
        if(disp is not None):
            txt = ' ' + txt + ' '
            img = img.adaptiveScale(disp.resolution)
            layer = DrawingLayer((img.width,img.height))
            layer.setFontSize(60)
            layer.ezViewText(txt,(20,20),fgcolor=color)
            img.addDrawingLayer(layer)
            img.applyLayers()
            img.save(disp)
