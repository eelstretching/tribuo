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

package org.tribuo.classification.sgd.linear;

import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.provenance.ModelProvenance;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The inference time version of a linear model trained using SGD.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends AbstractLinearSGDModel<Label> {
    private static final long serialVersionUID = 2L;

    private final VectorNormalizer normalizer;

    LinearSGDModel(String name, ModelProvenance provenance,
                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo,
                   LinearParameters parameters, VectorNormalizer normalizer, boolean generatesProbabilities) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters.getWeightMatrix(), generatesProbabilities);
        this.normalizer = normalizer;
    }

    private LinearSGDModel(String name, ModelProvenance provenance,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo,
                          DenseMatrix weights, VectorNormalizer normalizer, boolean generatesProbabilities) {
        super(name, provenance, featureIDMap, outputIDInfo, weights, generatesProbabilities);
        this.normalizer = normalizer;
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        PredAndActive predTuple = predictSingle(example);
        DenseVector prediction = predTuple.prediction;
        prediction.normalize(normalizer);

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> predMap = new LinkedHashMap<>();
        for (int i = 0; i < prediction.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabel();
            double score = prediction.get(i);
            Label label = new Label(labelName, score);
            predMap.put(labelName,label);
            if (score > maxScore) {
                maxScore = score;
                maxLabel = label;
            }
        }
        return new Prediction<>(maxLabel, predMap, predTuple.numActiveFeatures-1, example, generatesProbabilities);
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,newProvenance,featureIDMap,outputIDInfo,new DenseMatrix(weights),normalizer,generatesProbabilities);
    }

    @Override
    protected String getDimensionName(int index) {
        return outputIDInfo.getOutput(index).getLabel();
    }
}
