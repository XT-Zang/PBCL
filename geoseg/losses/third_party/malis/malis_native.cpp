#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

namespace {

struct DisjointSet64 {
    explicit DisjointSet64(std::size_t size) : parent(size), rank(size, 0) {
        std::iota(parent.begin(), parent.end(), std::uint64_t{0});
    }

    std::uint64_t find(std::uint64_t value) {
        std::uint64_t root = value;
        while (parent[root] != root) root = parent[root];
        while (parent[value] != value) {
            const auto next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    }

    std::pair<std::uint64_t, std::uint64_t> unite(
        std::uint64_t first, std::uint64_t second) {
        first = find(first);
        second = find(second);
        if (first == second) return {first, second};
        if (rank[first] < rank[second]) std::swap(first, second);
        parent[second] = first;
        if (rank[first] == rank[second]) ++rank[first];
        return {first, second};
    }

    std::vector<std::uint64_t> parent;
    std::vector<std::uint8_t> rank;
};

struct EdgeItem {
    std::uint32_t key;
    std::uint32_t edge;
};

struct Entry {
    std::uint64_t label;
    std::uint64_t count;
};

struct Bag32 {
    std::uint64_t single_label;
    std::uint32_t total;
    std::uint32_t kind_or_pool;  // 0=empty, 1=single, >=2 => pool_idx = kind_or_pool - 2
};

struct Workspace {
    std::vector<std::uint32_t> parent;
    std::vector<std::uint8_t> rank;
    std::vector<Bag32> bags;
    std::vector<EdgeItem> items;
    std::vector<EdgeItem> scratch;
    std::vector<std::vector<Entry>> pools;
    std::vector<Entry> merge_buf;
    std::size_t active_pools = 0;

    std::vector<std::uint8_t> known_buf;
    std::vector<float> constrained_buf;
    std::vector<std::uint64_t> counts_buf;

    inline std::uint32_t find(std::uint32_t value) {
        std::uint32_t root = value;
        while (parent[root] != root) root = parent[root];
        while (parent[value] != value) {
            const auto next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    }

    inline std::pair<std::uint32_t, std::uint32_t> unite(
        std::uint32_t first, std::uint32_t second) {
        if (rank[first] < rank[second]) std::swap(first, second);
        parent[second] = first;
        if (rank[first] == rank[second]) ++rank[first];
        return {first, second};
    }

    inline std::uint32_t alloc_pool() {
        if (active_pools == pools.size()) {
            pools.emplace_back();
            pools.back().reserve(16);
        } else {
            pools[active_pools].clear();
        }
        return static_cast<std::uint32_t>(active_pools++);
    }
};

inline std::uint32_t float_desc_key(float w) {
    if (w == 0.0f) w = 0.0f;
    std::uint32_t bits;
    std::memcpy(&bits, &w, sizeof(bits));
    const std::uint32_t asc = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return ~asc;
}

inline std::uint64_t lookup_sorted(const std::vector<Entry>& vec, std::uint64_t label) {
    std::size_t lo = 0, hi = vec.size();
    while (lo < hi) {
        const std::size_t mid = lo + ((hi - lo) >> 1);
        if (vec[mid].label < label) lo = mid + 1;
        else hi = mid;
    }
    return (lo < vec.size() && vec[lo].label == label) ? vec[lo].count : 0;
}

inline void insert_sorted(std::vector<Entry>& vec, std::uint64_t label, std::uint64_t count) {
    std::size_t lo = 0, hi = vec.size();
    while (lo < hi) {
        const std::size_t mid = lo + ((hi - lo) >> 1);
        if (vec[mid].label < label) lo = mid + 1;
        else hi = mid;
    }
    if (lo < vec.size() && vec[lo].label == label) {
        vec[lo].count += count;
    } else {
        vec.insert(vec.begin() + static_cast<std::ptrdiff_t>(lo), Entry{label, count});
    }
}

void malis_weights_64_fallback(
    const std::uint64_t vertex_count,
    const std::uint64_t* segmentation,
    const std::uint64_t edge_count,
    const std::uint64_t* node_first,
    const std::uint64_t* node_second,
    const float* edge_weight,
    const int positive_pass,
    std::uint64_t* pair_count_per_edge) {
    std::vector<std::map<std::uint64_t, std::uint64_t>> overlap(vertex_count);
    for (std::uint64_t index = 0; index < vertex_count; ++index) {
        if (segmentation[index] != 0) overlap[index][segmentation[index]] = 1;
    }

    std::vector<std::uint64_t> order;
    order.reserve(edge_count);
    for (std::uint64_t edge = 0; edge < edge_count; ++edge) {
        if (node_first[edge] < vertex_count && node_second[edge] < vertex_count) {
            order.push_back(edge);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](auto first, auto second) {
        return edge_weight[first] > edge_weight[second];
    });

    DisjointSet64 sets(vertex_count);
    for (const auto edge : order) {
        const auto first_root = sets.find(node_first[edge]);
        const auto second_root = sets.find(node_second[edge]);
        if (first_root == second_root) continue;

        std::uint64_t pair_count = 0;
        for (const auto& first : overlap[first_root]) {
            for (const auto& second : overlap[second_root]) {
                const bool selected = positive_pass
                    ? first.first == second.first
                    : first.first != second.first;
                if (selected) pair_count += first.second * second.second;
            }
        }
        pair_count_per_edge[edge] += pair_count;

        const auto roots = sets.unite(first_root, second_root);
        auto keep = roots.first;
        auto discard = roots.second;
        for (const auto& item : overlap[discard]) overlap[keep][item.first] += item.second;
        overlap[discard].clear();
    }
}

void radix_sort_edges(Workspace& ws) {
    ws.scratch.resize(ws.items.size());
    std::size_t counts[4][256] = {};
    for (const auto& it : ws.items) {
        counts[0][it.key & 0xFFu]++;
        counts[1][(it.key >> 8) & 0xFFu]++;
        counts[2][(it.key >> 16) & 0xFFu]++;
        counts[3][(it.key >> 24) & 0xFFu]++;
    }
    for (int pass = 0; pass < 4; ++pass) {
        std::size_t offsets[256];
        std::size_t sum = 0;
        for (int b = 0; b < 256; ++b) {
            offsets[b] = sum;
            sum += counts[pass][b];
        }
        const int shift = pass * 8;
        for (const auto& it : ws.items) {
            ws.scratch[offsets[(it.key >> shift) & 0xFFu]++] = it;
        }
        ws.items.swap(ws.scratch);
    }
}

inline void merge_edge_mst(
    Workspace& ws,
    const std::uint32_t edge,
    const std::uint32_t u_raw,
    const std::uint32_t v_raw,
    const int positive_pass,
    std::uint64_t* pair_count_per_edge) {
    const std::uint32_t u = ws.find(u_raw);
    const std::uint32_t v = ws.find(v_raw);
    if (u == v) return;

    Bag32& a = ws.bags[u];
    Bag32& b = ws.bags[v];
    const std::uint32_t ta = a.total;
    const std::uint32_t tb = b.total;

    if (ta != 0 && tb != 0) {
        std::uint64_t same_pairs = 0;
        const std::uint32_t ka = a.kind_or_pool;
        const std::uint32_t kb = b.kind_or_pool;
        if (ka == 1 && kb == 1) {
            if (a.single_label == b.single_label) {
                same_pairs = static_cast<std::uint64_t>(ta) * tb;
            }
        } else if (ka == 1) {
            same_pairs = static_cast<std::uint64_t>(ta) * lookup_sorted(ws.pools[kb - 2], a.single_label);
        } else if (kb == 1) {
            same_pairs = static_cast<std::uint64_t>(tb) * lookup_sorted(ws.pools[ka - 2], b.single_label);
        } else {
            const auto& va = ws.pools[ka - 2];
            const auto& vb = ws.pools[kb - 2];
            if (va.size() <= vb.size()) {
                for (const auto& e : va) same_pairs += e.count * lookup_sorted(vb, e.label);
            } else {
                for (const auto& e : vb) same_pairs += e.count * lookup_sorted(va, e.label);
            }
        }
        const std::uint64_t pair_count = positive_pass
            ? same_pairs
            : (static_cast<std::uint64_t>(ta) * tb - same_pairs);
        pair_count_per_edge[edge] += pair_count;
    }

    const auto roots = ws.unite(u, v);
    const std::uint32_t keep = roots.first;
    const std::uint32_t discard = roots.second;
    Bag32& bk = ws.bags[keep];
    Bag32& bd = ws.bags[discard];
    if (bd.kind_or_pool == 0) return;
    if (bk.kind_or_pool == 0) {
        bk = bd;
        bd.kind_or_pool = 0;
        bd.total = 0;
        return;
    }

    const std::uint32_t new_total = bk.total + bd.total;
    if (bk.kind_or_pool == 1 && bd.kind_or_pool == 1) {
        if (bk.single_label == bd.single_label) {
            bk.total = new_total;
        } else {
            const std::uint32_t pidx = ws.alloc_pool();
            auto& vec = ws.pools[pidx];
            if (bk.single_label < bd.single_label) {
                vec.push_back(Entry{bk.single_label, bk.total});
                vec.push_back(Entry{bd.single_label, bd.total});
            } else {
                vec.push_back(Entry{bd.single_label, bd.total});
                vec.push_back(Entry{bk.single_label, bk.total});
            }
            bk.kind_or_pool = pidx + 2;
            bk.total = new_total;
        }
    } else if (bk.kind_or_pool >= 2 && bd.kind_or_pool == 1) {
        insert_sorted(ws.pools[bk.kind_or_pool - 2], bd.single_label, bd.total);
        bk.total = new_total;
    } else if (bk.kind_or_pool == 1 && bd.kind_or_pool >= 2) {
        insert_sorted(ws.pools[bd.kind_or_pool - 2], bk.single_label, bk.total);
        bk = bd;
        bk.total = new_total;
    } else {
        if (ws.pools[bk.kind_or_pool - 2].size() < ws.pools[bd.kind_or_pool - 2].size()) {
            std::swap(bk.kind_or_pool, bd.kind_or_pool);
        }
        auto& dst = ws.pools[bk.kind_or_pool - 2];
        const auto& src = ws.pools[bd.kind_or_pool - 2];
        if (src.size() <= 4 && dst.size() >= 16) {
            for (const auto& e : src) insert_sorted(dst, e.label, e.count);
        } else {
            ws.merge_buf.clear();
            ws.merge_buf.reserve(dst.size() + src.size());
            std::size_t i = 0, j = 0;
            while (i < dst.size() && j < src.size()) {
                if (dst[i].label < src[j].label) ws.merge_buf.push_back(dst[i++]);
                else if (src[j].label < dst[i].label) ws.merge_buf.push_back(src[j++]);
                else {
                    ws.merge_buf.push_back(Entry{dst[i].label, dst[i].count + src[j].count});
                    ++i; ++j;
                }
            }
            while (i < dst.size()) ws.merge_buf.push_back(dst[i++]);
            while (j < src.size()) ws.merge_buf.push_back(src[j++]);
            dst.swap(ws.merge_buf);
        }
        ws.pools[bd.kind_or_pool - 2].clear();
        bk.total = new_total;
    }
    bd.kind_or_pool = 0;
    bd.total = 0;
}

void init_bags(Workspace& ws, const std::uint32_t n_verts, const std::uint64_t* segmentation) {
    ws.parent.resize(n_verts);
    std::iota(ws.parent.begin(), ws.parent.end(), std::uint32_t{0});
    ws.rank.assign(n_verts, 0);
    ws.bags.resize(n_verts);
    ws.active_pools = 0;
    for (std::uint32_t i = 0; i < n_verts; ++i) {
        const auto lbl = segmentation[i];
        if (lbl != 0) {
            ws.bags[i] = Bag32{lbl, 1, 1};
        } else {
            ws.bags[i].total = 0;
            ws.bags[i].kind_or_pool = 0;
        }
    }
}

}  // namespace

extern "C" void geoseg_malis_loss_weights(
    const std::uint64_t vertex_count,
    const std::uint64_t* segmentation,
    const std::uint64_t edge_count,
    const std::uint64_t* node_first,
    const std::uint64_t* node_second,
    const float* edge_weight,
    const int positive_pass,
    std::uint64_t* pair_count_per_edge) {

    if (vertex_count > 0xFFFFFFFFull || edge_count > 0xFFFFFFFFull) {
        malis_weights_64_fallback(
            vertex_count, segmentation, edge_count, node_first, node_second,
            edge_weight, positive_pass, pair_count_per_edge);
        return;
    }

    thread_local Workspace ws;
    ws.items.clear();
    if (ws.items.capacity() < edge_count) ws.items.reserve(edge_count);
    for (std::uint64_t e = 0; e < edge_count; ++e) {
        const auto u = node_first[e];
        const auto v = node_second[e];
        if (u < vertex_count && v < vertex_count) {
            const float w = edge_weight[e];
            if (!(w == w)) {
                malis_weights_64_fallback(
                    vertex_count, segmentation, edge_count, node_first, node_second,
                    edge_weight, positive_pass, pair_count_per_edge);
                return;
            }
            ws.items.push_back(EdgeItem{float_desc_key(w), static_cast<std::uint32_t>(e)});
        }
    }
    radix_sort_edges(ws);
    init_bags(ws, static_cast<std::uint32_t>(vertex_count), segmentation);

    for (const auto& it : ws.items) {
        const std::uint32_t edge = it.edge;
        merge_edge_mst(
            ws, edge,
            static_cast<std::uint32_t>(node_first[edge]),
            static_cast<std::uint32_t>(node_second[edge]),
            positive_pass, pair_count_per_edge);
    }
}

extern "C" void geoseg_constrained_malis_sample(
    const std::uint64_t height,
    const std::uint64_t width,
    const float* predicted_affinities,
    const std::uint64_t* instance_map,
    const std::uint8_t* valid_mask,
    float* out_targets,
    float* out_weights,
    std::uint64_t* out_pos_edges,
    std::uint64_t* out_neg_edges,
    int* out_active) {

    const std::uint64_t hw64 = height * width;
    if (height == 0 || width == 0 || hw64 > 0x7FFFFFFFull) {
        *out_pos_edges = 0;
        *out_neg_edges = 0;
        *out_active = 0;
        return;
    }

    const std::uint32_t H = static_cast<std::uint32_t>(height);
    const std::uint32_t W = static_cast<std::uint32_t>(width);
    const std::uint32_t HW = H * W;
    const std::uint32_t E = 2u * HW;

    thread_local Workspace ws;
    ws.known_buf.assign(E, 0);
    std::memset(out_targets, 0, sizeof(float) * E);
    std::memset(out_weights, 0, sizeof(float) * E);

    std::uint64_t first_lbl = 0;
    bool has_two_or_more = false;
    for (std::uint32_t i = 0; i < HW; ++i) {
        const std::uint64_t lbl = instance_map[i];
        if (lbl != 0) {
            if (first_lbl == 0) first_lbl = lbl;
            else if (lbl != first_lbl) { has_two_or_more = true; break; }
        }
    }

    for (std::uint32_t y = 0; y < H; ++y) {
        const std::uint32_t row = y * W;
        for (std::uint32_t x = 0; x + 1 < W; ++x) {
            const std::uint32_t idx = row + x;
            const std::uint64_t a = instance_map[idx];
            const std::uint64_t b = instance_map[idx + 1];
            const bool k = (a > 0 || b > 0) &&
                (valid_mask == nullptr || (valid_mask[idx] != 0 && valid_mask[idx + 1] != 0));
            ws.known_buf[idx] = k ? 1 : 0;
            out_targets[idx] = (a > 0 && a == b) ? 1.0f : 0.0f;
        }
    }
    for (std::uint32_t y = 0; y + 1 < H; ++y) {
        const std::uint32_t row = y * W;
        for (std::uint32_t x = 0; x < W; ++x) {
            const std::uint32_t idx = row + x;
            const std::uint32_t e = HW + idx;
            const std::uint64_t a = instance_map[idx];
            const std::uint64_t b = instance_map[idx + W];
            const bool k = (a > 0 || b > 0) &&
                (valid_mask == nullptr || (valid_mask[idx] != 0 && valid_mask[idx + W] != 0));
            ws.known_buf[e] = k ? 1 : 0;
            out_targets[e] = (a > 0 && a == b) ? 1.0f : 0.0f;
        }
    }

    if (first_lbl == 0) {
        *out_pos_edges = 0;
        *out_neg_edges = 0;
        *out_active = 0;
        return;
    }

    ws.constrained_buf.resize(E);
    ws.counts_buf.resize(E);

    std::uint64_t pos_nz = 0;
    std::uint64_t neg_nz = 0;

    for (int positive = 0; positive <= 1; ++positive) {
        if (positive == 0 && !has_two_or_more) continue;
        const float not_pos = positive ? 0.0f : 1.0f;

        ws.items.clear();
        if (ws.items.capacity() < E) ws.items.reserve(E);
        bool has_nan = false;

        for (std::uint32_t y = 0; y < H; ++y) {
            const std::uint32_t row = y * W;
            for (std::uint32_t x = 0; x + 1 < W; ++x) {
                const std::uint32_t e = row + x;
                const float w = (ws.known_buf[e] && out_targets[e] == not_pos)
                    ? not_pos
                    : predicted_affinities[e];
                if (!(w == w)) { has_nan = true; break; }
                ws.items.push_back(EdgeItem{float_desc_key(w), e});
            }
            if (has_nan) break;
        }
        if (!has_nan) {
            for (std::uint32_t y = 0; y + 1 < H; ++y) {
                const std::uint32_t row = y * W;
                for (std::uint32_t x = 0; x < W; ++x) {
                    const std::uint32_t e = HW + row + x;
                    const float w = (ws.known_buf[e] && out_targets[e] == not_pos)
                        ? not_pos
                        : predicted_affinities[e];
                    if (!(w == w)) { has_nan = true; break; }
                    ws.items.push_back(EdgeItem{float_desc_key(w), e});
                }
                if (has_nan) break;
            }
        }

        std::memset(ws.counts_buf.data(), 0, sizeof(std::uint64_t) * E);
        if (has_nan) {
            // NaN fallback path matching std::stable_sort behavior
            std::vector<std::uint64_t> n1(E), n2(E, ~std::uint64_t{0});
            std::vector<float> cw(E);
            for (std::uint32_t e = 0; e < E; ++e) {
                cw[e] = (ws.known_buf[e] && out_targets[e] == not_pos) ? not_pos : predicted_affinities[e];
            }
            for (std::uint32_t y = 0; y < H; ++y) {
                for (std::uint32_t x = 0; x < W; ++x) {
                    const std::uint64_t idx = static_cast<std::uint64_t>(y) * W + x;
                    n1[idx] = idx;
                    n1[HW + idx] = idx;
                    if (x + 1 < W) n2[idx] = idx + 1;
                    if (y + 1 < H) n2[HW + idx] = idx + W;
                }
            }
            malis_weights_64_fallback(HW, instance_map, E, n1.data(), n2.data(), cw.data(), positive, ws.counts_buf.data());
        } else {
            radix_sort_edges(ws);
            init_bags(ws, HW, instance_map);
            for (const auto& it : ws.items) {
                const std::uint32_t edge = it.edge;
                const std::uint32_t u_raw = (edge < HW) ? edge : (edge - HW);
                const std::uint32_t v_raw = (edge < HW) ? (edge + 1) : (edge - HW + W);
                merge_edge_mst(ws, edge, u_raw, v_raw, positive, ws.counts_buf.data());
            }
        }

        double total = 0.0;
        for (std::uint32_t e = 0; e < E; ++e) {
            if (ws.known_buf[e] && out_targets[e] != not_pos) {
                const float val = static_cast<float>(ws.counts_buf[e]);
                ws.constrained_buf[e] = val;
                total += static_cast<double>(val);
            } else {
                ws.constrained_buf[e] = 0.0f;
            }
        }
        if (total > 0.0) {
            const float denom = static_cast<float>(total);
            std::uint64_t nz = 0;
            for (std::uint32_t e = 0; e < E; ++e) {
                const float v = ws.constrained_buf[e] / denom;
                if (v != 0.0f) {
                    out_weights[e] += v;
                    ++nz;
                }
            }
            if (positive) pos_nz = nz;
            else neg_nz = nz;
        }
    }

    int any_w = 0;
    for (std::uint32_t e = 0; e < E; ++e) {
        if (out_weights[e] != 0.0f) { any_w = 1; break; }
    }
    *out_pos_edges = pos_nz;
    *out_neg_edges = neg_nz;
    *out_active = any_w;
}
