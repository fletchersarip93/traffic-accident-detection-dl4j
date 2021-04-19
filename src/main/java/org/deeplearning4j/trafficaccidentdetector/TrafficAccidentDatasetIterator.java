package org.deeplearning4j.trafficaccidentdetector;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.codec.reader.NativeCodecRecordReader;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public class TrafficAccidentDatasetIterator {

    private int videoWidth;
    private int videoHeight;

    public TrafficAccidentDatasetIterator(int videoWidth, int videoHeight) {
        this.videoWidth = videoWidth;
        this.videoHeight = videoHeight;
    }

    public DataSetIterator getDataSetIterator(String dataDirectory, String labelDirectory, int startIdx, int nExamples, int miniBatchSize, int nFrames) throws Exception {
        //Here, our data and labels are in separate files
        //videos: 000000.mp4, 000001.mp4, etc
        //labels: 000000.csv, 000001.csv, etc. One time step per line

        SequenceRecordReader featuresTrain = getFeaturesReader(dataDirectory, startIdx, nExamples, nFrames);
        SequenceRecordReader labelsTrain = getLabelsReader(labelDirectory, startIdx, nExamples);

        SequenceRecordReaderDataSetIterator sequenceIter =
                new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 1, true);
        sequenceIter.setPreProcessor(new VideoPreProcessor());

        //AsyncDataSetIterator: Used to (pre-load) load data in a separate thread
        return new AsyncDataSetIterator(sequenceIter,1);
//        return sequenceIter;
    }

    public SequenceRecordReader getFeaturesReader(String path, int startIdx, int num, int nFrames) throws IOException, InterruptedException {
        //InputSplit is used here to define what the file paths look like
        InputSplit is = new NumberedFileInputSplit(path + "%d.mp4", startIdx, startIdx + num - 1);

        Configuration conf = new Configuration();
        conf.set(NativeCodecRecordReader.RAVEL, "true");
        conf.set(NativeCodecRecordReader.START_FRAME, "0");
        conf.set(NativeCodecRecordReader.TOTAL_FRAMES, String.valueOf(nFrames));
        conf.set(NativeCodecRecordReader.ROWS, String.valueOf(videoHeight));
        conf.set(NativeCodecRecordReader.COLUMNS, String.valueOf(videoWidth));
        NativeCodecRecordReader crr = new NativeCodecRecordReader();
        crr.initialize(conf, is);
        return crr;

    }

    public SequenceRecordReader getLabelsReader(String path, int startIdx, int num) throws Exception {
        InputSplit isLabels = new NumberedFileInputSplit(path + "%d.csv", startIdx, startIdx + num - 1);
        CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
        csvSeq.initialize(isLabels);
        return csvSeq;
    }

    public static class VideoPreProcessor implements DataSetPreProcessor {
        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
            toPreProcess.getFeatures().divi(255);  //[0,255] -> [0,1] for input pixel values
        }
    }

}
