/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tribuo.multilabel.baseline;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.StopWatch;
import java.io.IOException;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.multilabel.ImmutableMultiLabelInfo;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Trains n independent binary {@link Model}s, each of which predicts a single
 * {@link Label}.
 * <p>
 * Then wraps it up in an {@link IndependentMultiLabelModel} to provide a
 * {@link MultiLabel} prediction.
 * <p>
 * It trains each model sequentially, and could be optimised to train in
 * parallel.
 */
public class ThreadedIndependentMultiLabelTrainer implements Trainer<MultiLabel> {

    private static final Logger logger = Logger.getLogger(ThreadedIndependentMultiLabelTrainer.class.getName());

    @Config(mandatory = true, description = "Trainer to use for each individual label.")
    private Trainer<Label> innerTrainer;

    @Config(description = "Size of core thread pool")
    private int corePoolSize = 8;

    @Config(description = "Maximim size of thread pool")
    private int maxPoolSize = 16;

    private int trainInvocationCounter = 0;

    /**
     * for olcut.
     */
    private ThreadedIndependentMultiLabelTrainer() {
    }

    public ThreadedIndependentMultiLabelTrainer(Trainer<Label> innerTrainer) {
        this.innerTrainer = innerTrainer;
    }

    @Override
    public Model<MultiLabel> train(Dataset<MultiLabel> examples, Map<String, Provenance> runProvenance) {
        logger.log(Level.FINE, "Starting multi-label trainng");
        BlockingQueue<Runnable> workQueue = new LinkedBlockingQueue<>(20);
        ThreadPoolExecutor executor = new ThreadPoolExecutor(corePoolSize, maxPoolSize, 10, TimeUnit.SECONDS, workQueue);
        if(examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        ImmutableMultiLabelInfo labelInfo = (ImmutableMultiLabelInfo) examples.getOutputIDInfo();
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        List<Label> labelList = new ArrayList<>();
        DatasetProvenance datasetProvenance = examples.getProvenance();
        List<Future<Model<Label>>> futures = new ArrayList<>();
        for(MultiLabel l : labelInfo.getDomain()) {

            Label label = new Label(l.getLabelString());
            labelList.add(label);
            logger.info(String.format("Adding callable for %s", label.getLabel()));
            try {
                logger.info(String.format("Building training data for %s", label.getLabel()));
                MutableDataset<Label> trainingData = new MutableDataset<>(datasetProvenance, new LabelFactory());
                for(Example<MultiLabel> e : examples) {
                    Label newLabel = e.getOutput().createLabel(label);
                    // This sets the label in the new example to either l or MultiLabel.NEGATIVE_LABEL_STRING.
                    trainingData.add(new BinaryExample(e, newLabel));
                }

                futures.add(executor.submit(() -> {
                    StopWatch sw = new StopWatch();
                    sw.start();
                    Model<Label> model = innerTrainer.train(trainingData);
                    sw.stop();
                    logger.info(String.format("Training %s took %,dms", label.getLabel(), sw.getTimeMillis()));
                    return model;
                }));
            } catch(RejectedExecutionException ex) {
                logger.info(String.format("%s rejected?", label.getLabel()));
            }
        }

        //
        // Collect our futures
        List<Model<Label>> modelsList = new ArrayList<>();
        for(Future<Model<Label>> future : futures) {
            try {
                modelsList.add(future.get());
            } catch(InterruptedException ex) {
                logger.log(Level.SEVERE, "Error getting model results", ex);
            } catch(ExecutionException ex) {
                logger.log(Level.SEVERE, "Error getting model results", ex);
            }
        }
        executor.shutdown();
        ModelProvenance provenance = new ModelProvenance(IndependentMultiLabelModel.class.getName(), OffsetDateTime.now(), datasetProvenance, getProvenance(), runProvenance);
        trainInvocationCounter++;
        return new IndependentMultiLabelModel(labelList, modelsList, provenance, featureMap, labelInfo);
    }
    
    @Override
    public void postConfig() throws PropertyException, IOException {
        Trainer.super.postConfig();
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCounter;
    }

    @Override
    public String toString() {
        return "IndependentMultiLabelTrainer(innerTrainer=" + innerTrainer.toString() + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
