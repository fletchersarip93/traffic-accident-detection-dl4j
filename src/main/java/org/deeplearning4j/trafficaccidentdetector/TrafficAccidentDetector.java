package org.deeplearning4j.trafficaccidentdetector;

import org.apache.commons.io.FileUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.codec.reader.NativeCodecRecordReader;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.zoo.model.ResNet50;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Map;

public class TrafficAccidentDetector {

    public static final int N_VIDEOS = 500; // TODO: to be determined
    public static final int V_WIDTH = 224;
    public static final int V_HEIGHT = 224;
    public static final int V_NFRAMES = 50; // TODO: to be determined, may be computed along the way

    // Note that you will need to run with at least 7G off heap memory
    // if you want to keep this batchsize and train the nn config specified
    private static final int miniBatchSize = 2;
    private static final int seed = 1234;
    private static final double learningRate = 0.04;

    public static void main(String[] args) throws Exception {
        String dataDirectory = new ClassPathResource("CarAccidentData/OriData/videos").getPath();
        String labelDirectory = new ClassPathResource("CarAccidentData/OriData/videos_labels").getPath();

        //Set up network architecture:
        // use ResNet50
        ZooModel zooModel = ResNet50.builder().build();
        ComputationGraph resnet50 = (ComputationGraph) zooModel.initPretrained();
        System.out.println(resnet50.summary());

        FineTuneConfiguration fineTuneConfig = new FineTuneConfiguration.Builder()
                .seed(seed)
                .l2(0.001) //l2 regularization on all layers
                .updater(new AdaGrad(learningRate))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.LEAKYRELU)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10)
                .backpropType(BackpropType.TruncatedBPTT)
                .tbpttFwdLength(V_NFRAMES / 5)
                .tbpttBackLength(V_NFRAMES / 5)
                .build();

        ComputationGraph resnet50Transfer = new TransferLearning.GraphBuilder(resnet50)
                .fineTuneConfiguration(fineTuneConfig)
                .setFeatureExtractor("flatten_1")
                .addLayer("lstm1", new LSTM.Builder()
                        .activation(Activation.TANH)
                        .nIn(1000)
                        .nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new AdaGrad(0.008))
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(),
                        new FeedForwardToRnnPreProcessor(),
                        "fc1000")
                .addLayer("rnnOutput", new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(100)
                        .nOut(1)    // either an accident or not
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .setOutputs("rnnOutput")
                .build();
        System.out.println(resnet50Transfer.summary());
//
//
//
//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(12345)
//                .l2(0.001) //l2 regularization on all layers
//                .updater(new AdaGrad(0.04))
//                .list()
//                .layer(new ConvolutionLayer.Builder(10, 10)
//                        .nIn(3) //3 channels: RGB
//                        .nOut(30)
//                        .stride(4, 4)
//                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.RELU)
//                        .build())   //Output: (130-10+0)/4+1 = 31 -> 31*31*30
//                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//                        .kernelSize(3, 3)
//                        .stride(2, 2).build())   //(31-3+0)/2+1 = 15
//                .layer(new ConvolutionLayer.Builder(3, 3)
//                        .nIn(30)
//                        .nOut(10)
//                        .stride(2, 2)
//                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.RELU)
//                        .build())   //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
//                .layer(new DenseLayer.Builder()
//                        .activation(Activation.RELU)
//                        .nIn(490)
//                        .nOut(50)
//                        .weightInit(WeightInit.RELU)
//                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                        .gradientNormalizationThreshold(10)
//                        .updater(new AdaGrad(0.01))
//                        .build())
//                .layer(new LSTM.Builder()
//                        .activation(Activation.TANH)
//                        .nIn(50)
//                        .nOut(50)
//                        .weightInit(WeightInit.XAVIER)
//                        .updater(new AdaGrad(0.008))
//                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                        .gradientNormalizationThreshold(10)
//                        .build())
//                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//                        .activation(Activation.SOFTMAX)
//                        .nIn(50)
//                        .nOut(4)    //4 possible shapes: circle, square, arc, line
//                        .weightInit(WeightInit.XAVIER)
//                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                        .gradientNormalizationThreshold(10)
//                        .build())
//                .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
//                .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 10))
//                .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
//                .backpropType(BackpropType.TruncatedBPTT)
//                .tBPTTForwardLength(V_NFRAMES / 5)
//                .tBPTTBackwardLength(V_NFRAMES / 5)
//                .build();
//
//        MultiLayerNetwork net = new MultiLayerNetwork(conf);
//        net.init();
//
//        // summary of layer and parameters
//        System.out.println(net.summary());
//
//        int testStartIdx = (int) (0.9 * N_VIDEOS);  //90% in train, 10% in test
//        int nTest = N_VIDEOS - testStartIdx;
//
//        //Conduct learning
//        System.out.println("Starting training...");
//        net.setListeners(new ScoreIterationListener(1));
//
//        int nTrainEpochs = 15;
//        for (int i = 0; i < nTrainEpochs; i++) {
//            DataSetIterator trainData = getDataSetIterator(dataDirectory, 0, testStartIdx - 1, miniBatchSize);
//            while(trainData.hasNext())
//                net.fit(trainData.next());
//            Nd4j.saveBinary(net.params(),new File("videomodel.bin"));
//            FileUtils.writeStringToFile(new File("videoconf.json"), conf.toJson(), (Charset) null);
//            System.out.println("Epoch " + i + " complete");
//
//            //Evaluate classification performance:
//            evaluatePerformance(net,testStartIdx,nTest,dataDirectory);
//        }
    }


    private static void evaluatePerformance(MultiLayerNetwork net, int testStartIdx, int nExamples, String outputDirectory) throws Exception {
        //Assuming here that the full test data set doesn't fit in memory -> load 10 examples at a time
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "circle");
        labelMap.put(1, "square");
        labelMap.put(2, "arc");
        labelMap.put(3, "line");
        Evaluation evaluation = new Evaluation(labelMap);

        DataSetIterator testData = getDataSetIterator(outputDirectory, testStartIdx, nExamples, 10);
        while(testData.hasNext()) {
            DataSet dsTest = testData.next();
            INDArray predicted = net.output(dsTest.getFeatures(), false);
            INDArray actual = dsTest.getLabels();
            evaluation.evalTimeSeries(actual, predicted);
        }

        System.out.println(evaluation.stats());
    }

    private static DataSetIterator getDataSetIterator(String dataDirectory, int startIdx, int nExamples, int miniBatchSize) throws Exception {
        //Here, our data and labels are in separate files
        //videos: shapes_0.mp4, shapes_1.mp4, etc
        //labels: shapes_0.txt, shapes_1.txt, etc. One time step per line

        SequenceRecordReader featuresTrain = getFeaturesReader(dataDirectory, startIdx, nExamples);
        SequenceRecordReader labelsTrain = getLabelsReader(dataDirectory, startIdx, nExamples);

        SequenceRecordReaderDataSetIterator sequenceIter =
                new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 4, false);
        sequenceIter.setPreProcessor(new VideoPreProcessor());

        //AsyncDataSetIterator: Used to (pre-load) load data in a separate thread
        return new AsyncDataSetIterator(sequenceIter,1);
    }

    private static SequenceRecordReader getFeaturesReader(String path, int startIdx, int num) throws IOException, InterruptedException {
        //InputSplit is used here to define what the file paths look like
        InputSplit is = new NumberedFileInputSplit(path + "shapes_%d.mp4", startIdx, startIdx + num - 1);

        Configuration conf = new Configuration();
        conf.set(NativeCodecRecordReader.RAVEL, "true");
        conf.set(NativeCodecRecordReader.START_FRAME, "0");
        conf.set(NativeCodecRecordReader.TOTAL_FRAMES, String.valueOf(V_NFRAMES));
        conf.set(NativeCodecRecordReader.ROWS, String.valueOf(V_WIDTH));
        conf.set(NativeCodecRecordReader.COLUMNS, String.valueOf(V_HEIGHT));
        NativeCodecRecordReader crr = new NativeCodecRecordReader();
        crr.initialize(conf, is);
        return crr;

    }

    private static SequenceRecordReader getLabelsReader(String path, int startIdx, int num) throws Exception {
        InputSplit isLabels = new NumberedFileInputSplit(path + "shapes_%d.txt", startIdx, startIdx + num - 1);
        CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
        csvSeq.initialize(isLabels);
        return csvSeq;
    }

    private static class VideoPreProcessor implements DataSetPreProcessor {
        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
            toPreProcess.getFeatures().divi(255);  //[0,255] -> [0,1] for input pixel values
        }
    }

}
