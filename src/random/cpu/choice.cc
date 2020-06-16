/*!
 *  Copyright (c) 2019 by Contributors
 * \file random/choice.cc
 * \brief Non-uniform discrete sampling implementation
 */

#include <dgl/random.h>
#include <dgl/array.h>
#include <vector>
#include <numeric>
#include "sample_utils.h"

namespace dgl {

template<typename IdxType>
IdxType RandomEngine::Choice(FloatArray prob) {
  IdxType ret = 0;
  ATEN_FLOAT_TYPE_SWITCH(prob->dtype, ValueType, "probability", {
    // TODO(minjie): allow choosing different sampling algorithms
    utils::TreeSampler<IdxType, ValueType, true> sampler(this, prob);
    ret = sampler.Draw();
  });
  return ret;
}

template int32_t RandomEngine::Choice<int32_t>(FloatArray);
template int64_t RandomEngine::Choice<int64_t>(FloatArray);


template<typename IdxType, typename FloatType>
void RandomEngine::Choice(IdxType num, FloatArray prob, IdxType* out, bool replace) {
  const IdxType N = prob->shape[0];
  if (!replace)
    CHECK_LE(num, N) << "Cannot take more sample than population when 'replace=false'";
  if (num == N && !replace)
    std::iota(out, out + num, 0);

  utils::BaseSampler<IdxType>* sampler = nullptr;
  if (replace) {
    sampler = new utils::TreeSampler<IdxType, FloatType, true>(this, prob);
  } else {
    sampler = new utils::TreeSampler<IdxType, FloatType, false>(this, prob);
  }
  for (IdxType i = 0; i < num; ++i)
    out[i] = sampler->Draw();
  delete sampler;
}

template void RandomEngine::Choice<int32_t, float>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, float>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, double>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, double>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);

template <typename IdxType>
void RandomEngine::UniformChoice(IdxType num, IdxType population, IdxType* out, bool replace) {
  if (!replace)
    CHECK_LE(num, population) << "Cannot take more sample than population when 'replace=false'";
  if (replace) {
    for (IdxType i = 0; i < num; ++i)
      out[i] = RandInt(population);
  } else {
    if (num < population / 10) {  // TODO(minjie): may need a better threshold here
      // use hash set
      // In the best scenario, time complexity is O(num), i.e., no conflict.
      //
      // Let k be num / population, the expected number of extra sampling steps is roughly
      // k^2 / (1-k) * population, which means in the worst case scenario,
      // the time complexity is O(population^2). In practice, we use 1/10 since
      // std::unordered_set is pretty slow.
      std::unordered_set<IdxType> selected;
      while (selected.size() < num) {
        selected.insert(RandInt(population));
      }
      std::copy(selected.begin(), selected.end(), out);
    } else {
      // reservoir algorithm
      // time: O(population), space: O(num)
      for (IdxType i = 0; i < num; ++i)
        out[i] = i;
      for (IdxType i = num; i < population; ++i) {
        const IdxType j = RandInt(i);
        if (j < num)
          out[j] = i;
      }
    }
  }
}

template void RandomEngine::UniformChoice<int32_t>(
    int32_t num, int32_t population, int32_t* out, bool replace);
template void RandomEngine::UniformChoice<int64_t>(
    int64_t num, int64_t population, int64_t* out, bool replace);

template <typename IdxType, typename FloatType>
IdArray RandomEngine::BiasedChoice(IdxType num, IdArray split, FloatArray bias, IdxType* out, bool replace) {
  // get probability of each tag
  int64_t num_tags = bias->shape[0];
  std::vector<FloatType> prob(num_tags);
  FloatType *bias_data = static_cast<FloatType *>(bias->data);
  int64_t *split_data = static_cast<int64_t *>(split->data);
  int64_t total_node_num = 0;
  for (int64_t tag = 0 ; tag < num_tags; ++tag) {
    int64_t tag_num_nodes = split_data[tag+1] - split_data[tag];
    total_node_num += tag_num_nodes;
    FloatType tag_bias = bias_data[tag];
    prob[tag] = tag_num_nodes * tag_bias;
    // std::cerr << tag << " " << tag_bias << " "  << tag_num_nodes << " " << prob[tag] << std::endl;
  }

  auto tree = utils::TreeSampler<IdxType, FloatType, false>(this, NDArray::FromVector(prob));
  assert(total_node_num >= num);
  // we use hash set here. Maybe in the future we should support reservoir algorithm
  std::vector<std::unordered_set<IdxType>> selected(num_tags);
  for (int64_t i = 0 ; i < num ; ++i) {
    // first choose a tag
    IdxType tag = tree.Draw(&bias); 
    // then choose a node
    bool inserted = false;
    int64_t tag_num_nodes = split_data[tag+1] - split_data[tag];
    IdxType selected_node;
    while (!inserted) {
      CHECK_LT(selected[tag].size(), tag_num_nodes) << "To few nodes to be sampled";
      selected_node = RandInt(tag_num_nodes);
      inserted = selected[tag].insert(selected_node).second;
    }
    out[i] = selected_node + split_data[tag];
  }
}

template IdArray RandomEngine::BiasedChoice<int32_t, float>(int32_t, IdArray, FloatArray, int32_t*, bool );
template IdArray RandomEngine::BiasedChoice<int32_t, double>(int32_t, IdArray, FloatArray, int32_t*, bool );
template IdArray RandomEngine::BiasedChoice<int64_t, float>(int64_t, IdArray, FloatArray, int64_t*, bool );
template IdArray RandomEngine::BiasedChoice<int64_t, double>(int64_t, IdArray, FloatArray, int64_t*, bool );

};  // namespace dgl
