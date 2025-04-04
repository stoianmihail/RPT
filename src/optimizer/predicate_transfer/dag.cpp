#include "duckdb/optimizer/predicate_transfer/dag.hpp"

namespace duckdb {
    void DAGNode::AddIn(idx_t from, Expression* filter, bool forward) {
        if(forward) {
            for (auto &node : forward_in_) {
                if(node->GetDest() == from) {
                    node->Push(filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(from);
            node->Push(filter);
            forward_in_.emplace_back(std::move(node));
        } else {
            for (auto &node : backward_in_) {
                if(node->GetDest() == from) {
                    node->Push(filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(from);
            node->Push(filter);
            backward_in_.emplace_back(std::move(node));
        }
        return;
    }

#ifdef UseHashFilter
    void DAGNode::AddIn(idx_t from, shared_ptr<HashFilter> bloom_filter, bool forward) { 
#else
    void DAGNode::AddIn(idx_t from, shared_ptr<BlockedBloomFilter> bloom_filter, bool forward) {
#endif
        if(forward) {
            for (auto &node : forward_in_) {
                if(node->GetDest() == from) {
                    node->Push(bloom_filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(from);
            node->Push(bloom_filter);
            forward_in_.emplace_back(std::move(node));
        } else {
            for (auto &node : backward_in_) {
                if(node->GetDest() == from) {
                    node->Push(bloom_filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(from);
            node->Push(bloom_filter);
            backward_in_.emplace_back(std::move(node));
        }
        return;
    }

    void DAGNode::AddOut(idx_t to, Expression* filter, bool forward) {
        if(forward) {
            for (auto &node : forward_out_) {
                if(node->GetDest() == to) {
                    node->Push(filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(to);
            node->Push(std::move(filter));
            forward_out_.emplace_back(std::move(node));
        } else {
            for (auto &node : backward_out_) {
                if(node->GetDest() == to) {
                    node->Push(filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(to);
            node->Push(std::move(filter));
            backward_out_.emplace_back(std::move(node));
        }
        return;
    }

#ifdef UseHashFilter
    void DAGNode::AddOut(idx_t to, shared_ptr<HashFilter> bloom_filter, bool forward) {
#else
    void DAGNode::AddOut(idx_t to, shared_ptr<BlockedBloomFilter> bloom_filter, bool forward) {
#endif
        if(forward) {
            for (auto &node : forward_out_) {
                if(node->GetDest() == to) {
                    node->Push(bloom_filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(to);
            node->Push(bloom_filter);
            forward_out_.emplace_back(std::move(node));
        } else {
            for (auto &node : backward_out_) {
                if(node->GetDest() == to) {
                    node->Push(bloom_filter);
                    return;
                }
            }
            auto node = make_uniq<DAGEdge>(to);
            node->Push(bloom_filter);
            backward_out_.emplace_back(std::move(node));
        }
        return;
    }
}