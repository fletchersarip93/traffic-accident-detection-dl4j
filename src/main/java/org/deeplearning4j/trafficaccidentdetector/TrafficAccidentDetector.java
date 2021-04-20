package org.deeplearning4j.trafficaccidentdetector;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrafficAccidentDetector {

    private static final int trainMaxIndex = 130;
    private static final int testMaxIndex = 154;

    public static final int V_WIDTH = 130;
    public static final int V_HEIGHT = 130;
    public static final int nChannels = 3;

    // Note that you will need to run with at least 7G off heap memory
    // if you want to keep this batchsize and train the nn config specified
    private static final int miniBatchSize = 1;
//    private static final int seed = 560;
    private static final double learningRate = 0.00001;
    private static final double l2Coeff = 0.0001;
    private static final int truncatedBPTTLength = 50;
    private static final int nTrainEpochs = 300;
    private static final int FRAME_LENGTH = 361;

    public static void main(String[] args) throws Exception {
        String dataDirectory = "src/main/resources/CarAccidentData/OriData/videos/";
        String labelDirectory = "src/main/resources/CarAccidentData/OriData/videos_labels/";

        MultiLayerNetwork net = (MultiLayerNetwork) getFromScratchModel();
//        ComputationGraph net = (ComputationGraph) getTransferLearningModel();

        //Conduct learning
        System.out.println("Starting training...");
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        net.setListeners(
                new ScoreIterationListener(1),
                new StatsListener(storage, 1));

        TrafficAccidentDatasetIterator trafficAccidentDatasetIterator = new TrafficAccidentDatasetIterator(V_WIDTH, V_HEIGHT);

//        List<Integer> trainingVideoIndexes = new ArrayList<>();
//        for (int i = 0; i < nTrainEpochs; i++) {
//            // loop over the training videos
//            for (Integer trainingVideoIndex : trainingVideoIndexes) {
//                int nFrames = 0; // TODO: get this video's number of frames
//                trafficAccidentDatasetIterator.getDataSetIterator(dataDirectory, labelDirectory, trainingVideoIndex, 1, 1, nFrames);
//            }
//        }



        for (int i = 0; i < nTrainEpochs; i++) {
            DataSetIterator trainData = trafficAccidentDatasetIterator.getDataSetIterator(dataDirectory, labelDirectory, 0, trainMaxIndex+1, miniBatchSize, FRAME_LENGTH);
            while(trainData.hasNext()) {
                System.out.println("@@@next batch");
                net.fit(trainData.next());
            }
            Nd4j.saveBinary(net.params(),new File("videomodel.bin"));
            FileUtils.writeStringToFile(new File("videoconf.json"), net.conf().toJson(), (Charset) null);
            System.out.println("Epoch " + i + " complete");

            //Evaluate classification performance:
            evaluatePerformance(trafficAccidentDatasetIterator, net, trainMaxIndex+1, testMaxIndex-trainMaxIndex, dataDirectory, labelDirectory, miniBatchSize);
        }
    }

    private static Model getFromScratchModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
                .l2(l2Coeff) //l2 regularization on all layers
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.LEAKYRELU)
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(3) //3 channels: RGB
                        .nOut(30)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.LEAKYRELU)
                        .build())   //Output: (130-3+2)/1+1 = 224 -> 130*130*30
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())   //(224-2+0)/2+1 = 112 -> 65*65*30
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(30)
                        .nOut(10)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.LEAKYRELU)
                        .build())   //Output: (112-3+2)/1+1 = 112 -> 65*65*10
                .layer(new DenseLayer.Builder()
                        .nIn(42250)
                        .nOut(50)
                        .activation(Activation.LEAKYRELU)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .layer(new LSTM.Builder()
                        .activation(Activation.LEAKYRELU)
                        .nIn(50)
                        .nOut(50)
                        .weightInitRecurrent(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(50)
                        .nOut(1)    // either accident or not
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, nChannels))
                .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(65, 65, 10))
                .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(truncatedBPTTLength)
                .tBPTTBackwardLength(truncatedBPTTLength)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());

        return model;
    }

    private static Model getTransferLearningModel() throws IOException {
        // use ResNet50
        ZooModel zooModel = ResNet50.builder().build();
        ComputationGraph resnet50 = (ComputationGraph) zooModel.initPretrained();
        System.out.println(resnet50.summary());

        FineTuneConfiguration fineTuneConfig = new FineTuneConfiguration.Builder()
//                .seed(seed)
                .l2(0.001) //l2 regularization on all layers
                .updater(new AdaGrad(learningRate))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.LEAKYRELU)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10)
                .backpropType(BackpropType.TruncatedBPTT)
                .tbpttFwdLength(truncatedBPTTLength)
                .tbpttBackLength(truncatedBPTTLength)
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
                                .build(),
                        "lstm1")
                .setOutputs("rnnOutput")
                .build();
        resnet50Transfer.getConfiguration().addPreProcessors(InputType.convolutional(V_HEIGHT, V_WIDTH, nChannels));

        System.out.println(resnet50Transfer.summary());

        return resnet50Transfer;
    }

    private static void evaluatePerformance(TrafficAccidentDatasetIterator trafficAccidentDatasetIterator, MultiLayerNetwork net, int testStartIdx, int nExamples, String outputDirectory, String outputLabelDirectory, int miniBatchSize) throws Exception {
        //Assuming here that the full test data set doesn't fit in memory -> load 10 examples at a time
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "no_accident");
        labelMap.put(1, "accident");
        Evaluation evaluation = new Evaluation(labelMap);

        DataSetIterator testData = trafficAccidentDatasetIterator.getDataSetIterator(outputDirectory, outputLabelDirectory, testStartIdx, nExamples, miniBatchSize, FRAME_LENGTH);

        while(testData.hasNext()) {
            DataSet dsTest = testData.next();
            INDArray predicted = net.output(dsTest.getFeatures(), false);
            System.out.println(predicted);
            evaluation.evalTimeSeries(dsTest.getLabels(), predicted);
        }

        System.out.println(evaluation.stats());
    }

    private static void evaluatePerformance(TrafficAccidentDatasetIterator trafficAccidentDatasetIterator, ComputationGraph net, int testStartIdx, int nExamples, String outputDirectory, String outputLabelDirectory, int miniBatchSize) throws Exception {
        //Assuming here that the full test data set doesn't fit in memory -> load 10 examples at a time
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "no_accident");
        labelMap.put(1, "accident");
        Evaluation evaluation = new Evaluation(labelMap);

        DataSetIterator testData = trafficAccidentDatasetIterator.getDataSetIterator(outputDirectory, outputLabelDirectory, testStartIdx, nExamples, miniBatchSize, FRAME_LENGTH);

        INDArray[] output = net.output(testData);
        System.out.println(output);

//        while(testData.hasNext()) {
//            DataSet dsTest = testData.next();
//            INDArray predicted = net.output(false, dsTest.getFeatures());
//            System.out.println(predicted);
//            evaluation.evalTimeSeries(dsTest.getLabels(), predicted);
//        }

        System.out.println(evaluation.stats());
    }

}
