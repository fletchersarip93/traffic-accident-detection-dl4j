package org.deeplearning4j.CNN_ViewNDArray;


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CNN_ViewNDArray_TrainSet {
    private static String[] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static double trainPerc = 1.0;
    private static Random rng = new Random();
    private static PathLabelGenerator labelMaker = new ParentPathLabelGenerator();  //Name of folder becomes name of label

    private static int width = 5; //200
    private static int height = 5; //200
    private static int nChannels = 3;

    private static int batchSize = 4;
    private static int nEpoch = 20;
    private static int seed = 123;
    private static int nClasses = 2;
    private static double learningRate = 0.001;

    public static void main(String[] args) throws Exception {
        File inputFile = new ClassPathResource("CNN_ViewNDArray").getFile();

        FileSplit split = new FileSplit(inputFile, allowedExt);

        PathFilter pathFilter = new BalancedPathFilter(rng, allowedExt, labelMaker);
        //BalancedPathFilter - Basically repeatedly take one sample from each class until one of the class runs out.

        InputSplit[] allData = split.sample(pathFilter, trainPerc, 1 - trainPerc);
        InputSplit trainData = allData[0];

        //Where is image augmentation
        //What about different image sizes

        ImageTransform hFlip = new FlipImageTransform(1);  //1 is horizontal, 0 is vertical, -1 is both
        ImageTransform rotate1 = new RotateImageTransform(15);
        ImageTransform rotate2 = new RotateImageTransform(-15);
        ImageTransform rCrop = new CropImageTransform(20);             //Remove x pixel from the Top and Bottom and Left and Right.
        //RandomCropTransform rCrop = new RandomCropTransform(60,60);  //You want the size of the cropped image to be exactly 60 x 60

        ImageTransform show = new ShowImageTransform("Augmentation");   //This allow you to see the Augmented images

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(hFlip, 0.0),
                new Pair<>(rotate1, 0.0),
                new Pair<>(rotate2, 0.0),
                new Pair<>(rCrop, 0.0),
                new Pair<>(show, 1.0)    // This means only 1% of the augmented image will be shown when running
        );

        PipelineImageTransform transform = new PipelineImageTransform(pipeline, false);

        //The height and width is to set the size of the image to be fed into the CNN
        //The ImageRecordReader will automatically resize your image to the argument's height and width.
        //So if your image is rectangle, but the argument is square. Your image will be squeezed to fit the square.
        ImageRecordReader trainRR = new ImageRecordReader(height, width, nChannels);

        trainRR.initialize(trainData, transform);   //Only apply augmentation on train dataset

        System.out.println("IMAGE ----------------------- ");
        System.out.println( trainRR.next() );
        System.out.println("IMAGE ----------------------- ");
        System.out.println( trainRR.next() );
        System.out.println("IMAGE ----------------------- ");
        System.out.println( trainRR.next() );
        System.out.println("IMAGE ----------------------- ");
        System.out.println( trainRR.next() );
        System.out.println("IMAGE ----------------------- ");
        System.out.println( trainRR.next() );
        System.out.println("DONE ----------------------- ");

        /*
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, nClasses);

        DataNormalization scaler = new ImagePreProcessingScaler();   // Scale Pixel range 0 to 1.
        trainIter.setPreProcessor(scaler);

        */

        //System.out.println(trainIter.next().getFeatures() );

        boolean test = true;
        while(test){
            Thread.sleep(1000);
        }

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .nIn(nChannels)
                        .nOut(16)    //Number of Filters
                        .activation(Activation.RELU)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .setInputType(InputType.convolutional(height, width, nChannels))   //This line is used so you no need to specify .nIn() in every layer
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        //model.fit(trainIter, nEpoch);

        //Evaluation evalTrain = model.evaluate(trainIter);

        //System.out.println("Train Evaluation:\n" + evalTrain.stats());
    }
}
