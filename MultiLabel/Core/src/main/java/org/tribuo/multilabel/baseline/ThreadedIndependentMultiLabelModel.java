package org.tribuo.multilabel.baseline;

import java.util.List;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.classification.Label;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.provenance.ModelProvenance;

/**
 *
 */
public class ThreadedIndependentMultiLabelModel extends IndependentMultiLabelModel {

    public ThreadedIndependentMultiLabelModel(List<Label> labels, List<Model<Label>> models, ModelProvenance description, ImmutableFeatureMap featureMap, ImmutableOutputInfo<MultiLabel> labelInfo) {
        super(labels, models, description, featureMap, labelInfo);
    }

}
