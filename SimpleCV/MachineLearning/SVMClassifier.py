from SimpleCV.base import *
from SimpleCV.ImageClass import Image, ImageSet
from SimpleCV.DrawingLayer import *
from SimpleCV.Features import FeatureExtractorBase
"""
This class is encapsulates almost everything needed to train, test, and deploy a
multiclass support vector machine for an image classifier. Training data should
be stored in separate directories for each class. This class uses the feature
extractor base class to  convert images into a feature vector. The basic workflow
is as follows.
1. Get data.
2. Setup Feature Extractors (roll your own or use the ones I have written).
3. Train the classifier.
4. Test the classifier.
5. Tweak parameters as necessary.
6. Repeat until you reach the desired accuracy.
7. Save the classifier.
8. Deploy using the classify method.
"""
class SVMClassifier(Classifier):
    """
    This class encapsulates a Naive Bayes Classifier.
    See:
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    mClassNames = []
    mDataSetRaw = []
    mDataSetOrange = []
    mClassifier = None
    mFeatureExtractors = None
    mOrangeDomain = None
    mSVMPrototype = None

    mKernelType = {}

    mSVMType = {}

    mSVMProperties = {
        'KernelType':'RBF', #default is a RBF Kernel
        'SVMType':'NU',     #default is C
        'nu':None,          # NU for SVM NU
        'c':None,           #C for SVM C - the slack variable
        'degree':None,      #degree for poly kernels - defaults to 3
        'coef':None,        #coef for Poly/Sigmoid defaults to 0
        'gamma':None,       #kernel param for poly/rbf/sigma - default is 1/#samples
    }
    #human readable to CV constant property mapping

    def __init__(self,featureExtractors,properties=None):

        if not ORANGE_ENABLED:
            logger.warning("The required orange machine learning library is not installed")
            return None

        self.mKernelType = {
            'RBF':orange.SVMLearner.RBF, #Radial basis kernel
            'Linear':orange.SVMLearner.Linear, #Linear basis kernel
            'Poly':orange.SVMLearner.Polynomial, #Polynomial kernel
            'Sigmoid':orange.SVMLearner.Sigmoid #Sigmoid Kernel
        }

        self.mSVMType = {
            'NU':orange.SVMLearner.Nu_SVC,
            'C':orange.SVMLearner.C_SVC
        }

        self.mFeatureExtractors =  featureExtractors
        if(properties is not None):
            self.mSVMProperties = properties
        self._parameterizeKernel()
        self.mClassNames = []
        self.mDataSetRaw = []
        self.mDataSetOrange = []
        self.mClassifier = None
        self.mOrangeDomain = None

    def setProperties(self, properties):
        """
        Note that resetting the properties will reset the SVM and you will need
        to retrain.
        """
        if(properties is not None):
            self.mSVMProperties = properties
        self._parameterizeKernel()

    def _parameterizeKernel(self):
        #Set the parameters for our SVM
        self.mSVMPrototype = orange.SVMLearner()
        self.mSVMPrototype.svm_type = self.mSVMType[self.mSVMProperties["SVMType"]]
        self.mSVMPrototype.kernel_type = self.mKernelType[self.mSVMProperties["KernelType"]]
        if(self.mSVMProperties["nu"] is not None):
            self.mSVMPrototype.nu = self.mSVMProperties["nu"]
        if(self.mSVMProperties["c"] is not None):
            self.mSVMPrototype.C = self.mSVMProperties["c"]
        if(self.mSVMProperties["degree"]  is not None):
            self.mSVMPrototype.degree = self.mSVMProperties["degree"]
        if(self.mSVMProperties["coef"] is not None):
            self.mSVMPrototype.coef0 = self.mSVMProperties["coef"]
        if(self.mSVMProperties["gamma"] is not None):
            self.mSVMPrototype.gamma = self.mSVMProperties["gamma"]

    def classify(self, image):
        """
        Classify a single image. Takes in an image and returns the string
        of the classification.

        Make sure you haved loaded the feauture extractors and the training data.

        """
        featureVector = []
        for extractor in self.mFeatureExtractors: #get the features
            feats = extractor.extract(image)
            if( feats is not None ):
                featureVector.extend(feats)
        featureVector.extend([self.mClassNames[0]])
        test = orange.ExampleTable(self.mOrangeDomain,[featureVector])
        c = self.mClassifier(test[0]) #classify
        return str(c) #return to class name

    def train(self,images,classNames,disp=None,subset=-1,savedata=None,verbose=True):
        """
        Train the classifier.
        images paramater can take in a list of paths or a list of imagesets
        images - the order of the paths or imagesets must be in the same order as the class type

        - Note all image classes must be in seperate directories
        - The class names must also align to the directories

        disp - if display is a display we show images and class label,
        otherwise nothing is done.

        subset - if subset = -1 we use the whole dataset. If subset = # then we
        use min(#images,subset)

        savedata - if save data is None nothing is saved. If savedata is a file
        name we save the data to a tab delimited file.

        verbose - print confusion matrix and file names
        returns [%Correct %Incorrect Confusion_Matrix]
        """
        count = 0
        self.mClassNames = classNames
        # fore each class, get all of the data in the path and train
        for i in range(len(classNames)):
            if ( isinstance(images[i], str) ):
                count = count + self._trainPath(images[i],classNames[i],subset,disp,verbose)
            else:
                count = count + self._trainImageSet(images[i],classNames[i],subset,disp,verbose)

        colNames = []
        for extractor in self.mFeatureExtractors:
            colNames.extend(extractor.getFieldNames())

        if(count <= 0):
            logger.warning("No features extracted - bailing")
            return None

        # push our data into an orange example table
        self.mOrangeDomain = orange.Domain(map(orange.FloatVariable,colNames),orange.EnumVariable("type",values=self.mClassNames))
        self.mDataSetOrange = orange.ExampleTable(self.mOrangeDomain,self.mDataSetRaw)
        if(savedata is not None):
            orange.saveTabDelimited (savedata, self.mDataSetOrange)

        self.mClassifier = self.mSVMPrototype(self.mDataSetOrange)
        correct = 0
        incorrect = 0
        for i in range(count):
            c = self.mClassifier(self.mDataSetOrange[i])
            test = self.mDataSetOrange[i].getclass()
            if verbose:
                print "original", test, "classified as", c
            if(test==c):
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        good = 100*(float(correct)/float(count))
        bad = 100*(float(incorrect)/float(count))

        confusion = 0
        if( len(self.mClassNames) > 2 ):
            crossValidator = orngTest.learnAndTestOnLearnData([self.mSVMPrototype],self.mDataSetOrange)
            confusion = orngStat.confusionMatrices(crossValidator)[0]

        if verbose:
            print("Correct: "+str(good))
            print("Incorrect: "+str(bad))
            classes = self.mDataSetOrange.domain.classVar.values
            print "\t"+"\t".join(classes)
            if confusion > 0:
                for className, classConfusions in zip(classes, confusion):
                   print ("%s" + ("\t%i" * len(classes))) % ((className, ) + tuple(classConfusions))

        return [good, bad, confusion]

    def test(self,images,classNames,disp=None,subset=-1,savedata=None,verbose=True):
        """
        Test the classifier.
        images paramater can take in a list of paths or a list of imagesets
        images - the order of the paths or imagesets must be in the same order as the class type

        - Note all image classes must be in seperate directories
        - The class names must also align to the directories

        disp - if display is a display we show images and class label,
        otherwise nothing is done.

        subset - if subset = -1 we use the whole dataset. If subset = # then we
        use min(#images,subset)

        savedata - if save data is None nothing is saved. If savedata is a file
        name we save the data to a tab delimited file.

        verbose - print confusion matrix and file names
        returns [%Correct %Incorrect Confusion_Matrix]
        """
        count = 0
        correct = 0
        self.mClassNames = classNames
        colNames = []
        for extractor in self.mFeatureExtractors:
            colNames.extend(extractor.getFieldNames())
            self.mOrangeDomain = orange.Domain(map(orange.FloatVariable,colNames),orange.EnumVariable("type",values=self.mClassNames))

        dataset = []
        for i in range(len(classNames)):
            if ( isinstance(images[i],str) ):
                [dataset,cnt,crct] =self._testPath(images[i],classNames[i],dataset,subset,disp,verbose)
                count = count + cnt
                correct = correct + crct
            else:
                [dataset,cnt,crct] =self._testImageSet(images[i],classNames[i],dataset,subset,disp,verbose)
                count = count + cnt
                correct = correct + crct


        testData = orange.ExampleTable(self.mOrangeDomain,dataset)

        if savedata is not None:
            orange.saveTabDelimited (savedata, testdata)

        confusion = 0
        if( len(self.mClassNames) > 2 ):
            crossValidator = orngTest.learnAndTestOnTestData([self.mSVMPrototype],self.mDataSetOrange,testData)
            confusion = orngStat.confusionMatrices(crossValidator)[0]

        good = 100*(float(correct)/float(count))
        bad = 100*(float(count-correct)/float(count))
        if verbose:
            print("Correct: "+str(good))
            print("Incorrect: "+str(bad))
            classes = self.mDataSetOrange.domain.classVar.values
            print "\t"+"\t".join(classes)
            for className, classConfusions in zip(classes, confusion):
                print ("%s" + ("\t%i" * len(classes))) % ((className, ) + tuple(classConfusions))

        return [good, bad, confusion]
