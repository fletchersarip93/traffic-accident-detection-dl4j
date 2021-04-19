package org.deeplearning4j.XOR_CNN;

import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

public class XOR_CNN_ChannelXOR {

    private static int width = 5; //200
    private static int height = 1; //200
    private static int nChannels = 2;

    private static int nEpoch = 20;
    private static int seed = 123;
    private static double learningRate = 0.003;

    public static void main(String[] args) {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .setInputTypes(InputType.convolutional(height, width, nChannels))   //This line is used so you no need to specify .nIn() in every layer
                .addInputs("input")
                .addLayer("CNN1", new ConvolutionLayer.Builder()
                        .nOut(2)    //Number of Filters
                        .activation(Activation.SIGMOID)
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "input")
                .addLayer("output", new CnnLossLayer.Builder()
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build(), "CNN1")
                .setOutputs("output")
                .backpropType(BackpropType.Standard)
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        System.out.println(model.summary());

        INDArray testInput1 = Nd4j.create(new double[][][] {{
                {   10.0000,   20.0000,   30.0000},
                {   20.0000,   25.0000,   35.0000}}});
        System.out.println(model.output(testInput1));
    }
}
