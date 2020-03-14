// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"
#include <math.h>

namespace onnxruntime {
namespace ml {
namespace detail {

struct TreeNodeElementId {
  int tree_id;
  int node_id;
  bool operator==(const TreeNodeElementId& xyz) const {
    return (tree_id == xyz.tree_id) && (node_id == xyz.node_id);
  }
  bool operator<(const TreeNodeElementId& xyz) const {
    return ((tree_id < xyz.tree_id) || (tree_id == xyz.tree_id && node_id < xyz.node_id));
  }
};

template <typename T>
struct SparseValue {
  int64_t i;
  T value;
};

template <typename T>
struct ScoreValue {
  unsigned char has_score;
  T score;
  operator float() const { return has_score ? static_cast<float>(score) : 0; }
  ScoreValue<T>& operator=(float v) {
    this->score = v;
    return *this;
  }
};

enum MissingTrack {
  NONE,
  TRUE,
  FALSE
};

template <typename T>
struct TreeNodeElement {
  TreeNodeElementId id;
  int feature_id;
  T value;
  T hitrates;
  NODE_MODE mode;
  TreeNodeElement<T>* truenode;
  TreeNodeElement<T>* falsenode;
  MissingTrack missing_tracks;
  std::vector<SparseValue<T>> weights;

  bool is_not_leaf;
  bool is_missing_track_true;
};

template <typename ITYPE, typename OTYPE>
class TreeAggregator {
 protected:
  size_t n_trees_;
  int64_t n_targets_or_classes_;
  POST_EVAL_TRANSFORM post_transform_;
  const std::vector<OTYPE>& base_values_;
  OTYPE origin_;
  bool use_base_values_;

 public:
  TreeAggregator(size_t n_trees,
                 const int64_t& n_targets_or_classes,
                 POST_EVAL_TRANSFORM post_transform,
                 const std::vector<OTYPE>& base_values) : n_trees_(n_trees), n_targets_or_classes_(n_targets_or_classes), post_transform_(post_transform), base_values_(base_values) {
    origin_ = base_values_.size() == 1 ? base_values_[0] : 0.f;
    use_base_values_ = base_values_.size() == static_cast<size_t>(n_targets_or_classes_);
  }

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<OTYPE>& /*prediction*/, const TreeNodeElement<OTYPE>& /*root*/) const {}

  void MergePrediction1(ScoreValue<OTYPE>& /*prediction*/, ScoreValue<OTYPE>& /*prediction2*/) const {}

  void FinalizeScores1(OTYPE* Z, ScoreValue<OTYPE>& prediction, int64_t* /*Y*/) const {
    prediction.score = prediction.has_score ? (prediction.score + origin_) : origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? static_cast<OTYPE>(ComputeProbit(static_cast<float>(prediction.score))) : prediction.score;
  }

  // N outputs

  void ProcessTreeNodePrediction(std::vector<ScoreValue<OTYPE>>& /*predictions*/, const TreeNodeElement<OTYPE>& /*root*/) const {}

  void MergePrediction(std::vector<ScoreValue<OTYPE>>& /*predictions*/, const std::vector<ScoreValue<OTYPE>>& /*predictions2*/) const {}

  void FinalizeScores(std::vector<ScoreValue<OTYPE>>& predictions, OTYPE* Z, int add_second_class, int64_t*) const {
    ORT_ENFORCE(predictions.size() == (size_t)n_targets_or_classes_);
    OTYPE val;
    auto it = predictions.begin();
    for (int64_t jt = 0; jt < n_targets_or_classes_; ++jt, ++it) {
      val = use_base_values_ ? base_values_[jt] : 0.f;
      val += it->has_score ? it->score : 0;
      it->score = val;
    }
    write_scores(predictions, post_transform_, Z, add_second_class);
  }
};

/////////////
// regression
/////////////

template <typename ITYPE, typename OTYPE>
class TreeAggregatorSum : public TreeAggregator<ITYPE, OTYPE> {
 public:
  TreeAggregatorSum(size_t n_trees,
                    const int64_t& n_targets_or_classes,
                    POST_EVAL_TRANSFORM post_transform,
                    const std::vector<OTYPE>& base_values) : TreeAggregator<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                          post_transform, base_values) {}

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<OTYPE>& prediction, const TreeNodeElement<OTYPE>& root) const {
    prediction.score += root.weights[0].value;
  }

  void MergePrediction1(ScoreValue<OTYPE>& prediction, const ScoreValue<OTYPE>& prediction2) const {
    prediction.score += prediction2.score;
  }

  void FinalizeScores1(OTYPE* Z, ScoreValue<OTYPE>& prediction, int64_t* /*Y*/) const {
    prediction.score += this->origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? static_cast<OTYPE>(ComputeProbit(static_cast<float>(prediction.score))) : prediction.score;
  }

  // N outputs

  void ProcessTreeNodePrediction(std::vector<ScoreValue<OTYPE>>& predictions, const TreeNodeElement<OTYPE>& root) const {
    for (auto it = root.weights.cbegin(); it != root.weights.cend(); ++it) {
      ORT_ENFORCE(it->i < (int64_t)predictions.size());
      predictions[it->i].score += it->value;
      predictions[it->i].has_score = 1;
    }
  }

  void MergePrediction(std::vector<ScoreValue<OTYPE>>& predictions, const std::vector<ScoreValue<OTYPE>>& predictions2) const {
    ORT_ENFORCE(predictions.size() == predictions2.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score += predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }

  void FinalizeScores(std::vector<ScoreValue<OTYPE>>& predictions, OTYPE* Z, int add_second_class, int64_t*) const {
    auto it = predictions.begin();
    if (this->use_base_values_) {
      auto it2 = this->base_values_.cbegin();
      for (; it != predictions.end(); ++it, ++it2)
        it->score = it->score + *it2;
    }
    write_scores(predictions, this->post_transform_, Z, add_second_class);
  }
};

template <typename ITYPE, typename OTYPE>
class TreeAggregatorAverage : public TreeAggregatorSum<ITYPE, OTYPE> {
 public:
  TreeAggregatorAverage(size_t n_trees,
                        const int64_t& n_targets_or_classes,
                        POST_EVAL_TRANSFORM post_transform,
                        const std::vector<OTYPE>& base_values) : TreeAggregatorSum<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                                 post_transform, base_values) {}

  void FinalizeScores1(OTYPE* Z, ScoreValue<OTYPE>& prediction, int64_t* /*Y*/) const {
    prediction.score /= this->n_trees_;
    prediction.score += this->origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? static_cast<OTYPE>(ComputeProbit(static_cast<float>(prediction.score))) : prediction.score;
  }

  void FinalizeScores(std::vector<ScoreValue<OTYPE>>& predictions, OTYPE* Z, int add_second_class, int64_t*) const {
    if (this->use_base_values_) {
      ORT_ENFORCE(this->base_values_.size() == predictions.size());
      auto it = predictions.begin();
      auto it2 = this->base_values_.cbegin();
      for (; it != predictions.end(); ++it, ++it2)
        it->score = it->score / this->n_trees_ + *it2;
    } else {
      auto it = predictions.begin();
      for (; it != predictions.end(); ++it)
        it->score /= this->n_trees_;
    }
    write_scores(predictions, this->post_transform_, Z, add_second_class);
  }
};

template <typename ITYPE, typename OTYPE>
class TreeAggregatorMin : public TreeAggregator<ITYPE, OTYPE> {
 public:
  TreeAggregatorMin(size_t n_trees,
                    const int64_t& n_targets_or_classes,
                    POST_EVAL_TRANSFORM post_transform,
                    const std::vector<OTYPE>& base_values) : TreeAggregator<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                          post_transform, base_values) {}

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<OTYPE>& prediction, const TreeNodeElement<OTYPE>& root) const {
    prediction.score = (!(prediction.has_score) || root.weights[0].value < prediction.score)
                           ? root.weights[0].value
                           : prediction.score;
    prediction.has_score = 1;
  }

  void MergePrediction1(ScoreValue<OTYPE>& prediction, const ScoreValue<OTYPE>& prediction2) const {
    if (prediction2.has_score) {
      prediction.score = prediction.has_score && (prediction.score < prediction2.score)
                             ? prediction.score
                             : prediction2.score;
      prediction.has_score = 1;
    }
  }

  // N outputs

  void ProcessTreeNodePrediction(std::vector<ScoreValue<OTYPE>>& predictions, const TreeNodeElement<OTYPE>& root) const {
    for (auto it = root.weights.begin(); it != root.weights.end(); ++it) {
      predictions[it->i].score = (!predictions[it->i].has_score || it->value < predictions[it->i].score)
                                     ? it->value
                                     : predictions[it->i].score;
      predictions[it->i].has_score = 1;
    }
  }

  void MergePrediction(std::vector<ScoreValue<OTYPE>>& predictions, const std::vector<ScoreValue<OTYPE>>& predictions2) const {
    ORT_ENFORCE(predictions.size() == predictions2.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score = predictions[i].has_score && (predictions[i].score < predictions2[i].score)
                                   ? predictions[i].score
                                   : predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }
};

template <typename ITYPE, typename OTYPE>
class TreeAggregatorMax : public TreeAggregator<ITYPE, OTYPE> {
 public:
  TreeAggregatorMax<ITYPE, OTYPE>(size_t n_trees,
                                  const int64_t& n_targets_or_classes,
                                  POST_EVAL_TRANSFORM post_transform,
                                  const std::vector<OTYPE>& base_values) : TreeAggregator<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                                        post_transform, base_values) {}

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<OTYPE>& prediction, const TreeNodeElement<OTYPE>& root) const {
    prediction.score = (!(prediction.has_score) || root.weights[0].value > prediction.score)
                           ? root.weights[0].value
                           : prediction.score;
    prediction.has_score = 1;
  }

  void MergePrediction1(ScoreValue<OTYPE>& prediction, const ScoreValue<OTYPE>& prediction2) const {
    if (prediction2.has_score) {
      prediction.score = prediction.has_score && (prediction.score > prediction2.score)
                             ? prediction.score
                             : prediction2.score;
      prediction.has_score = 1;
    }
  }

  // N outputs

  void ProcessTreeNodePrediction(std::vector<ScoreValue<OTYPE>>& predictions, const TreeNodeElement<OTYPE>& root) const {
    for (auto it = root.weights.begin(); it != root.weights.end(); ++it) {
      predictions[it->i].score = (!predictions[it->i].has_score || it->value > predictions[it->i].score)
                                     ? it->value
                                     : predictions[it->i].score;
      predictions[it->i].has_score = 1;
    }
  }

  void MergePrediction(std::vector<ScoreValue<OTYPE>>& predictions, const std::vector<ScoreValue<OTYPE>>& predictions2) const {
    ORT_ENFORCE(predictions.size() == predictions2.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score = predictions[i].has_score && (predictions[i].score > predictions2[i].score)
                                   ? predictions[i].score
                                   : predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }
};

/////////////////
// classification
/////////////////

template <typename ITYPE, typename OTYPE>
class TreeAggregatorClassifier : public TreeAggregatorSum<ITYPE, OTYPE> {
 private:
  const std::vector<int64_t>& class_labels_;
  bool binary_case_;
  bool weights_are_all_positive_;
  int64_t positive_label_;
  int64_t negative_label_;

 public:
  TreeAggregatorClassifier(size_t n_trees,
                           const int64_t& n_targets_or_classes,
                           POST_EVAL_TRANSFORM post_transform,
                           const std::vector<OTYPE>& base_values,
                           const std::vector<int64_t>& class_labels,
                           bool binary_case,
                           bool weights_are_all_positive,
                           int64_t positive_label = 1,
                           int64_t negative_label = 0) : TreeAggregatorSum<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                         post_transform, base_values),
                                                         class_labels_(class_labels),
                                                         binary_case_(binary_case),
                                                         weights_are_all_positive_(weights_are_all_positive),
                                                         positive_label_(positive_label),
                                                         negative_label_(negative_label) {}

  void get_max_weight(const std::vector<ScoreValue<OTYPE>>& classes, int64_t& maxclass, OTYPE& maxweight) const {
    maxclass = -1;
    maxweight = 0;
    for (auto it = classes.cbegin(); it != classes.cend(); ++it) {
      if (it->has_score && (maxclass == -1 || it->score > maxweight)) {
        maxclass = (int64_t)(it - classes.cbegin());
        maxweight = it->score;
      }
    }
  }

  int64_t _set_score_binary(int& write_additional_scores, const std::vector<ScoreValue<OTYPE>>& classes) const {
    ORT_ENFORCE(classes.size() == 2 || classes.size() == 1);
    return (classes.size() == 2 && classes[1].has_score)
               ? _set_score_binary(write_additional_scores, classes[0].score, classes[0].has_score, classes[1].score, classes[1].has_score)
               : _set_score_binary(write_additional_scores, classes[0].score, classes[0].has_score, 0, 0);
  }

  int64_t _set_score_binary(int& write_additional_scores, OTYPE score0, unsigned char has_score0, OTYPE score1, unsigned char has_score1) const {
    OTYPE pos_weight = has_score1 ? score1 : (has_score0 ? score0 : 0);  // only 1 class
    if (binary_case_) {
      if (weights_are_all_positive_) {
        if (pos_weight > 0.5) {
          write_additional_scores = 0;
          return class_labels_[1];  // positive label
        } else {
          write_additional_scores = 1;
          return class_labels_[0];  // negative label
        }
      } else {
        if (pos_weight > 0) {
          write_additional_scores = 2;
          return class_labels_[1];  // positive label
        } else {
          write_additional_scores = 3;
          return class_labels_[0];  // negative label
        }
      }
    }
    return (pos_weight > 0)
               ? positive_label_   // positive label
               : negative_label_;  // negative label
  }

  // 1 output

  void FinalizeScores1(OTYPE* Z, ScoreValue<OTYPE>& prediction, int64_t* Y) const {
    std::vector<OTYPE> scores(2);
    unsigned char has_scores[2] = {1, 0};

    int write_additional_scores = -1;
    if (this->base_values_.size() == 2) {
      // add base_values
      scores[1] = this->base_values_[1] + prediction.score;
      scores[0] = -scores[1];
      //has_score = true;
      has_scores[1] = 1;
    } else if (this->base_values_.size() == 1) {
      // ONNX is vague about two classes and only one base_values.
      scores[0] = prediction.score + this->base_values_[0];
      //if (!has_scores[1])
      //scores.pop_back();
      scores[0] = prediction.score;
    } else if (this->base_values_.size() == 0) {
      //if (!has_score)
      //  scores.pop_back();
      scores[0] = prediction.score;
    }

    *Y = _set_score_binary(write_additional_scores, scores[0], has_scores[0], scores[1], has_scores[1]);
    write_scores(scores, this->post_transform_, Z, write_additional_scores);
  }

  // N outputs

  void FinalizeScores(std::vector<ScoreValue<OTYPE>>& predictions, OTYPE* Z, int /*add_second_class*/, int64_t* Y = 0) const {
    OTYPE maxweight = 0;
    int64_t maxclass = -1;

    int write_additional_scores = -1;
    if (this->n_targets_or_classes_ > 2) {
      // add base values
      for (int64_t k = 0, end = static_cast<int64_t>(this->base_values_.size()); k < end; ++k) {
        if (!predictions[k].has_score) {
          predictions[k].has_score = 1;
          predictions[k].score = this->base_values_[k];
        } else {
          predictions[k].score += this->base_values_[k];
        }
      }
      get_max_weight(predictions, maxclass, maxweight);
      *Y = class_labels_[maxclass];
    } else {  // binary case
      ORT_ENFORCE(predictions.size() == 2);
      if (this->base_values_.size() == 2) {
        // add base values
        if (predictions[1].has_score) {
          // base_value_[0] is not used.
          // It assumes base_value[0] == base_value[1] in this case.
          // The specification does not forbid it but does not
          // say what the output should be in that case.
          predictions[1].score = this->base_values_[1] + predictions[0].score;
          predictions[0].score = -predictions[1].score;
          predictions[1].has_score = 1;
        } else {
          // binary as multiclass
          predictions[1].score += this->base_values_[1];
          predictions[0].score += this->base_values_[0];
        }
      } else if (this->base_values_.size() == 1) {
        // ONNX is vague about two classes and only one base_values.
        predictions[0].score += this->base_values_[0];
        if (!predictions[1].has_score)
          predictions.pop_back();
      } else if (this->base_values_.size() == 0) {
        // ONNX is vague about two classes and only one base_values.
        if (!predictions[1].has_score)
          predictions.pop_back();
      }

      *Y = _set_score_binary(write_additional_scores, predictions);
    }
    write_scores(predictions, this->post_transform_, Z, write_additional_scores);
	if (predictions.size() == 1)
		predictions.resize(2);
  }
};

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
