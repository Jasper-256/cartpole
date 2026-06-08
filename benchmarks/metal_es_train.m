#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

enum {
    DEFAULT_NUM_PENDULUMS = 2,
    DEFAULT_NUM_ENVS = 2359296,
    DEFAULT_STEPS = 500,
    MAX_THREADGROUP_SIZE = 128,
    MAX_SCRATCH_BYTES = 28 * 1024,
};

typedef struct {
    uint32_t steps;
    uint32_t num_envs;
    uint32_t num_groups;
    uint32_t iteration;
    float sigma;
    float learning_rate;
} TrainParams;

static uint32_t feature_count(uint32_t num_pendulums) {
    return 6u + 9u * num_pendulums;
}

static uint32_t weight_count(uint32_t num_pendulums) {
    return feature_count(num_pendulums) + 1u;
}

static uint32_t threadgroup_size_for_weights(uint32_t weights) {
    uint32_t size = MAX_THREADGROUP_SIZE;
    while (size > 1u && (uint64_t)weights * (uint64_t)size * sizeof(float) > MAX_SCRATCH_BYTES) {
        size >>= 1u;
    }
    return size;
}

static double now_seconds(void) {
    static mach_timebase_info_data_t info;
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    return (double)mach_absolute_time() * (double)info.numer / (double)info.denom * 1e-9;
}

static NSString *kernelSource(uint32_t num_pendulums, uint32_t threadgroup_size) {
    uint32_t state_size = num_pendulums + 1u;
    uint32_t features = feature_count(num_pendulums);
    uint32_t weights = weight_count(num_pendulums);
    return [NSString stringWithFormat:
            @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "#define NUM_PENDULUMS %uu\n"
            "#define STATE_SIZE %uu\n"
            "#define FEATURE_COUNT %uu\n"
            "#define WEIGHT_COUNT %uu\n"
            "#define THREADGROUP_SIZE %uu\n"
            "struct TrainParams { uint steps; uint num_envs; uint num_groups; uint iteration; float sigma; float learning_rate; };\n"
            "constant float DT = 0.02f;\n"
            "constant float GRAVITY = 9.8f;\n"
            "constant float CART_MASS = 1.0f;\n"
            "constant float POLE_MASS = 0.08f;\n"
            "constant float POLE_LENGTH = 0.7f;\n"
            "constant float FORCE_MAG = 14.0f;\n"
            "constant float CART_FRICTION = 0.08f;\n"
            "constant float POLE_FRICTION = 0.015f;\n"
            "constant float X_THRESHOLD = 2.4f;\n"
            "inline uint hash_u32(uint x) {\n"
            "    x ^= x >> 16u;\n"
            "    x *= 0x7feb352du;\n"
            "    x ^= x >> 15u;\n"
            "    x *= 0x846ca68bu;\n"
            "    x ^= x >> 16u;\n"
            "    return x;\n"
            "}\n"
            "inline float rand01(uint seed) {\n"
            "    return float(hash_u32(seed)) * (1.0f / 4294967296.0f);\n"
            "}\n"
            "inline float eps_sign(uint env_id, uint weight_id, uint iteration) {\n"
            "    uint pair_id = env_id >> 1u;\n"
            "    float pair_sign = (env_id & 1u) ? -1.0f : 1.0f;\n"
            "    float weight_sign = (hash_u32(pair_id * 747796405u + weight_id * 2891336453u + iteration * 277803737u) & 1u) ? 1.0f : -1.0f;\n"
            "    return pair_sign * weight_sign;\n"
            "}\n"
            "inline float perturbed_weight(constant float *weights, constant TrainParams &params, uint env_id, uint weight_id) {\n"
            "    return weights[weight_id] + params.sigma * eps_sign(env_id, weight_id, params.iteration);\n"
            "}\n"
            "inline float wrap_angle(float theta) {\n"
            "    const float pi = 3.14159265358979323846f;\n"
            "    float wrapped = fmod(theta + pi, 2.0f * pi);\n"
            "    if (wrapped < 0.0f) wrapped += 2.0f * pi;\n"
            "    return wrapped - pi;\n"
            "}\n"
            "inline uint state_index(uint row, uint col) {\n"
            "    return row * STATE_SIZE + col;\n"
            "}\n"
            "inline uint distal_count(uint index) {\n"
            "    return NUM_PENDULUMS - index;\n"
            "}\n"
            "inline void add_feature(thread float &force_logit,\n"
            "                        constant float *weights,\n"
            "                        constant TrainParams &params,\n"
            "                        uint env_id,\n"
            "                        thread uint &feature_id,\n"
            "                        float value) {\n"
            "    force_logit += perturbed_weight(weights, params, env_id, feature_id) * value;\n"
            "    feature_id += 1u;\n"
            "}\n"
            "kernel void es_rollout(constant float *weights [[buffer(0)]],\n"
            "                       device float *partials [[buffer(1)]],\n"
            "                       constant TrainParams &params [[buffer(2)]],\n"
            "                       uint gid [[thread_position_in_grid]],\n"
            "                       uint lid [[thread_position_in_threadgroup]],\n"
            "                       uint group_id [[threadgroup_position_in_grid]],\n"
            "                       threadgroup float *scratch [[threadgroup(0)]]) {\n"
            "    float score = 0.0f;\n"
            "    if (gid < params.num_envs) {\n"
            "    uint pair_id = gid >> 1u;\n"
            "    float u0 = rand01(pair_id * 13u + params.iteration * 1315423911u);\n"
            "    float u1 = rand01(pair_id * 17u + params.iteration * 2654435761u);\n"
            "    float x = 0.06f * u0 - 0.03f;\n"
            "    float x_dot = 0.06f * u1 - 0.03f;\n"
            "    float theta[NUM_PENDULUMS];\n"
            "    float theta_dot[NUM_PENDULUMS];\n"
            "    for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "        float theta_draw = rand01(pair_id * (19u + 4u * i) + params.iteration * (2246822519u + 1019667398u * i));\n"
            "        float theta_dot_draw = rand01(pair_id * (29u + 6u * i) + params.iteration * (1181783491u + 374761393u * i));\n"
            "        if (i == 0u) theta_dot_draw = u1;\n"
            "        if (i == 1u) theta_dot_draw = 1.0f - u0;\n"
            "        theta[i] = wrap_angle(3.14159265358979323846f - 0.08f + 0.16f * theta_draw);\n"
            "        theta_dot[i] = -0.02f + 0.04f * theta_dot_draw;\n"
            "    }\n"
            "    for (uint step = 0u; step < params.steps; ++step) {\n"
            "        float sin_theta[NUM_PENDULUMS];\n"
            "        float cos_theta[NUM_PENDULUMS];\n"
            "        float previous_cos[NUM_PENDULUMS];\n"
            "        float normalized_theta_dot[NUM_PENDULUMS];\n"
            "        float energy_error[NUM_PENDULUMS];\n"
            "        float theta_gate_sum = 0.0f;\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "            sin_theta[i] = sin(theta[i]);\n"
            "            cos_theta[i] = cos(theta[i]);\n"
            "            previous_cos[i] = cos_theta[i];\n"
            "            normalized_theta_dot[i] = theta_dot[i] / 10.0f;\n"
            "            float theta_gate = theta[i] / 0.7f;\n"
            "            theta_gate_sum += theta_gate * theta_gate;\n"
            "            float target_energy = 2.0f * GRAVITY * POLE_LENGTH;\n"
            "            float link_energy = 0.5f * (POLE_LENGTH * theta_dot[i]) * (POLE_LENGTH * theta_dot[i]) + GRAVITY * POLE_LENGTH * (cos_theta[i] + 1.0f);\n"
            "            energy_error[i] = (link_energy - target_energy) / target_energy;\n"
            "        }\n"
            "        float upright_gate = exp(-0.5f * theta_gate_sum / float(NUM_PENDULUMS));\n"
            "        float swing_gate = 1.0f - upright_gate;\n"
            "        float norm_x = x / X_THRESHOLD;\n"
            "        float norm_x_dot = x_dot / 5.0f;\n"
            "        float force_logit = perturbed_weight(weights, params, gid, FEATURE_COUNT);\n"
            "        uint feature_id = 0u;\n"
            "        add_feature(force_logit, weights, params, gid, feature_id, swing_gate);\n"
            "        add_feature(force_logit, weights, params, gid, feature_id, upright_gate);\n"
            "        add_feature(force_logit, weights, params, gid, feature_id, norm_x * swing_gate);\n"
            "        add_feature(force_logit, weights, params, gid, feature_id, norm_x_dot * swing_gate);\n"
            "        add_feature(force_logit, weights, params, gid, feature_id, norm_x * upright_gate);\n"
            "        add_feature(force_logit, weights, params, gid, feature_id, norm_x_dot * upright_gate);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, swing_gate * sin_theta[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, swing_gate * cos_theta[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, swing_gate * normalized_theta_dot[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, swing_gate * normalized_theta_dot[i] * sin_theta[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, swing_gate * normalized_theta_dot[i] * cos_theta[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, swing_gate * energy_error[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, swing_gate * energy_error[i] * normalized_theta_dot[i] * cos_theta[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, upright_gate * sin_theta[i]);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) add_feature(force_logit, weights, params, gid, feature_id, upright_gate * normalized_theta_dot[i]);\n"
            "        float force = FORCE_MAG * tanh(force_logit);\n"
            "        float matrix[STATE_SIZE * STATE_SIZE];\n"
            "        float lower[STATE_SIZE * STATE_SIZE];\n"
            "        float rhs[STATE_SIZE];\n"
            "        float q_acc[STATE_SIZE];\n"
            "        float y[STATE_SIZE];\n"
            "        for (uint i = 0u; i < STATE_SIZE * STATE_SIZE; ++i) {\n"
            "            matrix[i] = 0.0f;\n"
            "            lower[i] = 0.0f;\n"
            "        }\n"
            "        matrix[state_index(0u, 0u)] = CART_MASS + float(NUM_PENDULUMS) * POLE_MASS;\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "            float coupling = POLE_MASS * POLE_LENGTH * float(distal_count(i)) * cos_theta[i];\n"
            "            matrix[state_index(0u, i + 1u)] = coupling;\n"
            "            matrix[state_index(i + 1u, 0u)] = coupling;\n"
            "        }\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "            for (uint j = 0u; j < NUM_PENDULUMS; ++j) {\n"
            "                uint distal = NUM_PENDULUMS - max(i, j);\n"
            "                matrix[state_index(i + 1u, j + 1u)] = POLE_MASS * POLE_LENGTH * POLE_LENGTH * float(distal) * cos(theta[i] - theta[j]);\n"
            "            }\n"
            "        }\n"
            "        float bias_0 = CART_FRICTION * x_dot;\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "            bias_0 += -POLE_MASS * POLE_LENGTH * float(distal_count(i)) * sin_theta[i] * theta_dot[i] * theta_dot[i];\n"
            "        }\n"
            "        rhs[0] = force - bias_0;\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "            float bias = POLE_FRICTION * theta_dot[i] - POLE_MASS * GRAVITY * POLE_LENGTH * float(distal_count(i)) * sin_theta[i];\n"
            "            for (uint j = 0u; j < NUM_PENDULUMS; ++j) {\n"
            "                uint distal = NUM_PENDULUMS - max(i, j);\n"
            "                bias += POLE_MASS * POLE_LENGTH * POLE_LENGTH * float(distal) * sin(theta[i] - theta[j]) * theta_dot[j] * theta_dot[j];\n"
            "            }\n"
            "            rhs[i + 1u] = -bias;\n"
            "        }\n"
            "        for (uint col = 0u; col < STATE_SIZE; ++col) {\n"
            "            float diagonal = matrix[state_index(col, col)];\n"
            "            for (uint k = 0u; k < col; ++k) {\n"
            "                float value = lower[state_index(col, k)];\n"
            "                diagonal -= value * value;\n"
            "            }\n"
            "            lower[state_index(col, col)] = sqrt(max(diagonal, 1.0e-7f));\n"
            "            for (uint row = col + 1u; row < STATE_SIZE; ++row) {\n"
            "                float value = matrix[state_index(row, col)];\n"
            "                for (uint k = 0u; k < col; ++k) {\n"
            "                    value -= lower[state_index(row, k)] * lower[state_index(col, k)];\n"
            "                }\n"
            "                lower[state_index(row, col)] = value / lower[state_index(col, col)];\n"
            "            }\n"
            "        }\n"
            "        for (uint row = 0u; row < STATE_SIZE; ++row) {\n"
            "            float value = rhs[row];\n"
            "            for (uint col = 0u; col < row; ++col) value -= lower[state_index(row, col)] * y[col];\n"
            "            y[row] = value / lower[state_index(row, row)];\n"
            "        }\n"
            "        for (uint reverse_row = 0u; reverse_row < STATE_SIZE; ++reverse_row) {\n"
            "            uint row = STATE_SIZE - 1u - reverse_row;\n"
            "            float value = y[row];\n"
            "            for (uint col = row + 1u; col < STATE_SIZE; ++col) value -= lower[state_index(col, row)] * q_acc[col];\n"
            "            q_acc[row] = value / lower[state_index(row, row)];\n"
            "        }\n"
            "        x_dot += DT * q_acc[0];\n"
            "        x += DT * x_dot;\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "            theta_dot[i] += DT * q_acc[i + 1u];\n"
            "            theta[i] = wrap_angle(theta[i] + DT * theta_dot[i]);\n"
            "        }\n"
            "        bool finite_state = isfinite(x) && isfinite(x_dot);\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) finite_state = finite_state && isfinite(theta[i]) && isfinite(theta_dot[i]);\n"
            "        if (abs(x) > X_THRESHOLD || !finite_state) {\n"
            "            score -= 1000.0f;\n"
            "            break;\n"
            "        }\n"
            "        float height_sum = 0.0f;\n"
            "        float log_height_sum = 0.0f;\n"
            "        float progress_sum = 0.0f;\n"
            "        float theta_dot_sq_sum = 0.0f;\n"
            "        float top_theta_dot_sq_sum = 0.0f;\n"
            "        float theta_sq_sum = 0.0f;\n"
            "        float bottom_motion_sum = 0.0f;\n"
            "        float energy_err_sq_sum = 0.0f;\n"
            "        bool stable = abs(x) <= 0.5f && abs(x_dot) <= 0.75f;\n"
            "        for (uint i = 0u; i < NUM_PENDULUMS; ++i) {\n"
            "            float cos_value = cos(theta[i]);\n"
            "            float link_height = 0.5f * (cos_value + 1.0f);\n"
            "            float theta_dot_sq = theta_dot[i] * theta_dot[i];\n"
            "            float target_energy = 2.0f * GRAVITY * POLE_LENGTH;\n"
            "            float link_energy = 0.5f * (POLE_LENGTH * theta_dot[i]) * (POLE_LENGTH * theta_dot[i]) + GRAVITY * POLE_LENGTH * (cos_value + 1.0f);\n"
            "            float err = (link_energy - target_energy) / target_energy;\n"
            "            height_sum += link_height;\n"
            "            log_height_sum += log(max(link_height, 1.0e-4f));\n"
            "            progress_sum += cos_value - previous_cos[i];\n"
            "            theta_dot_sq_sum += theta_dot_sq;\n"
            "            top_theta_dot_sq_sum += link_height * link_height * theta_dot_sq;\n"
            "            theta_sq_sum += theta[i] * theta[i];\n"
            "            bottom_motion_sum += tanh(abs(theta_dot[i]) / 2.0f);\n"
            "            energy_err_sq_sum += err * err;\n"
            "            stable = stable && abs(theta[i]) <= 0.20943951f && abs(theta_dot[i]) <= 1.0f;\n"
            "        }\n"
            "        float inv_n = 1.0f / float(NUM_PENDULUMS);\n"
            "        float height = height_sum * inv_n;\n"
            "        float chain_height = exp(log_height_sum * inv_n);\n"
            "        float height_reward = 0.5f * height + 0.5f * chain_height;\n"
            "        float controlled = chain_height * exp(-0.5f * theta_dot_sq_sum);\n"
            "        float angle_precision = exp(-0.5f * theta_sq_sum / (0.35f * 0.35f));\n"
            "        float top_hold = angle_precision * exp(-0.5f * theta_dot_sq_sum);\n"
            "        float progress = progress_sum * inv_n;\n"
            "        float centered = 1.0f - clamp((x / X_THRESHOLD) * (x / X_THRESHOLD), 0.0f, 1.0f);\n"
            "        float bottom_motion = (1.0f - height) * bottom_motion_sum * inv_n;\n"
            "        float bottom_quiet = (1.0f - height) * (1.0f - clamp(bottom_motion, 0.0f, 1.0f));\n"
            "        float energy_score = exp(-0.5f * energy_err_sq_sum);\n"
            "        float velocity_cost = 0.01f * x_dot * x_dot + 2.0f * (x / X_THRESHOLD) * (x / X_THRESHOLD) + 0.002f * theta_dot_sq_sum * inv_n + 0.25f * top_theta_dot_sq_sum * inv_n;\n"
            "        float action_cost = 0.01f * (force / FORCE_MAG) * (force / FORCE_MAG);\n"
            "        score += 8.0f * height_reward + 100.0f * controlled + 200.0f * top_hold + 0.75f * energy_score + 60.0f * progress + 3.0f * bottom_motion + 0.1f * centered + 2000.0f * (stable ? 1.0f : 0.0f) - 1.5f * bottom_quiet - velocity_cost - action_cost;\n"
            "    }\n"
            "    }\n"
            "    if (!isfinite(score)) score = -10000.0f;\n"
            "    score = clamp(score, -10000.0f, 100000.0f);\n"
            "    for (uint j = 0u; j < WEIGHT_COUNT; ++j) scratch[j * THREADGROUP_SIZE + lid] = score * eps_sign(gid, j, params.iteration);\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint stride = THREADGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {\n"
            "        if (lid < stride) {\n"
            "            for (uint j = 0u; j < WEIGHT_COUNT; ++j) scratch[j * THREADGROUP_SIZE + lid] += scratch[j * THREADGROUP_SIZE + lid + stride];\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (lid == 0u) {\n"
            "        for (uint j = 0u; j < WEIGHT_COUNT; ++j) partials[group_id * WEIGHT_COUNT + j] = scratch[j * THREADGROUP_SIZE];\n"
            "    }\n"
            "}\n"
            "kernel void reduce_gradients(const device float *partials [[buffer(0)]],\n"
            "                             device float *gradients [[buffer(1)]],\n"
            "                             constant TrainParams &params [[buffer(2)]],\n"
            "                             uint j [[thread_position_in_grid]]) {\n"
            "    if (j >= WEIGHT_COUNT) return;\n"
            "    float total = 0.0f;\n"
            "    for (uint group = 0u; group < params.num_groups; ++group) {\n"
            "        total += partials[group * WEIGHT_COUNT + j];\n"
            "    }\n"
            "    gradients[j] = isfinite(total) ? total : 0.0f;\n"
            "}\n"
            "kernel void update_weights(device float *weights [[buffer(0)]],\n"
            "                           device float *gradients [[buffer(1)]],\n"
            "                           device float *momentum [[buffer(2)]],\n"
            "                           device float *variance [[buffer(3)]],\n"
            "                           constant TrainParams &params [[buffer(4)]],\n"
            "                           uint j [[thread_position_in_grid]]) {\n"
            "    if (j >= WEIGHT_COUNT) return;\n"
            "    float gradient = isfinite(gradients[j]) ? gradients[j] : 0.0f;\n"
            "    gradient = gradient / (float(params.num_envs) * params.sigma);\n"
            "    gradient = clamp(gradient, -200.0f, 200.0f);\n"
            "    momentum[j] = 0.9f * momentum[j] + 0.1f * gradient;\n"
            "    variance[j] = 0.99f * variance[j] + 0.01f * gradient * gradient;\n"
            "    float update = momentum[j] / (sqrt(variance[j]) + 1.0e-3f);\n"
            "    weights[j] = clamp(weights[j] + params.learning_rate * update, -20.0f, 20.0f);\n"
            "}\n",
            num_pendulums,
            state_size,
            features,
            weights,
            threadgroup_size];
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
    if (state == nil) {
        fprintf(stderr, "pipeline %s error: %s\n", name.UTF8String, error.localizedDescription.UTF8String);
        exit(1);
    }
    return state;
}

static void initialize_weights(float *weights, uint32_t num_pendulums) {
    uint32_t features = feature_count(num_pendulums);
    uint32_t total_weights = weight_count(num_pendulums);
    memset(weights, 0, total_weights * sizeof(float));

    weights[4] = -0.378319f;
    weights[5] = -1.719792f;

    uint32_t upright_sin_start = 6u + 7u * num_pendulums;
    uint32_t upright_dot_start = 6u + 8u * num_pendulums;
    if (num_pendulums == 1u) {
        weights[upright_sin_start] = 30.0f;
        weights[upright_dot_start] = 3.0f;
    } else {
        for (uint32_t i = 0; i < num_pendulums; ++i) {
            float normalized_link = (float)(i + 1u) / (float)num_pendulums;
            weights[upright_sin_start + i] = 29.908883f - 42.042058f * normalized_link;
            weights[upright_dot_start + i] = 26.445228f - 46.445228f * normalized_link;
        }
    }
    weights[features] = 0.0f;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        uint32_t num_pendulums = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : DEFAULT_NUM_PENDULUMS;
        uint32_t num_envs = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : DEFAULT_NUM_ENVS;
        uint32_t steps = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : DEFAULT_STEPS;
        uint32_t iterations = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 1u;
        float sigma = argc > 5 ? strtof(argv[5], NULL) : 0.25f;
        float learning_rate = argc > 6 ? strtof(argv[6], NULL) : 0.3f;
        const char *output_path = argc > 7 ? argv[7] : NULL;

        if (num_pendulums < 1u) {
            fprintf(stderr, "num_pendulums must be at least 1\n");
            return 1;
        }
        if (num_envs < 1u) {
            fprintf(stderr, "num_envs must be at least 1\n");
            return 1;
        }

        uint32_t features = feature_count(num_pendulums);
        uint32_t weights_total = weight_count(num_pendulums);
        uint32_t threadgroup_size = threadgroup_size_for_weights(weights_total);
        uint32_t num_groups = (num_envs + threadgroup_size - 1u) / threadgroup_size;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kernelSource(num_pendulums, threadgroup_size) options:nil error:&error];
        if (library == nil) {
            fprintf(stderr, "library error: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> rollout = pipeline(device, library, @"es_rollout");
        id<MTLComputePipelineState> reduce = pipeline(device, library, @"reduce_gradients");
        id<MTLComputePipelineState> update = pipeline(device, library, @"update_weights");

        id<MTLBuffer> weights = [device newBufferWithLength:(size_t)weights_total * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> partials = [device newBufferWithLength:(size_t)num_groups * weights_total * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> gradients = [device newBufferWithLength:(size_t)weights_total * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> momentum = [device newBufferWithLength:(size_t)weights_total * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> variance = [device newBufferWithLength:(size_t)weights_total * sizeof(float) options:MTLResourceStorageModeShared];
        if (weights == nil || partials == nil || gradients == nil || momentum == nil || variance == nil) {
            fprintf(stderr, "buffer allocation failed\n");
            return 1;
        }
        initialize_weights((float *)weights.contents, num_pendulums);
        memset(momentum.contents, 0, (size_t)weights_total * sizeof(float));
        memset(variance.contents, 0, (size_t)weights_total * sizeof(float));

        id<MTLCommandQueue> queue = [device newCommandQueue];
        MTLSize env_threads = MTLSizeMake(num_groups * threadgroup_size, 1, 1);
        MTLSize env_group = MTLSizeMake(threadgroup_size, 1, 1);
        NSUInteger weight_group_width = weights_total < 256u ? weights_total : 256u;
        MTLSize weight_threads = MTLSizeMake(weights_total, 1, 1);
        MTLSize weight_group = MTLSizeMake(weight_group_width, 1, 1);
        NSUInteger scratch_bytes = (NSUInteger)weights_total * threadgroup_size * sizeof(float);

        double start = now_seconds();
        id<MTLCommandBuffer> last_command = nil;
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
            TrainParams params = {
                .steps = steps,
                .num_envs = num_envs,
                .num_groups = num_groups,
                .iteration = iteration,
                .sigma = sigma,
                .learning_rate = learning_rate,
            };
            id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = [queue commandBuffer];

            id<MTLComputeCommandEncoder> rollout_encoder = [command computeCommandEncoder];
            [rollout_encoder setComputePipelineState:rollout];
            [rollout_encoder setBuffer:weights offset:0 atIndex:0];
            [rollout_encoder setBuffer:partials offset:0 atIndex:1];
            [rollout_encoder setBuffer:params_buffer offset:0 atIndex:2];
            [rollout_encoder setThreadgroupMemoryLength:scratch_bytes atIndex:0];
            [rollout_encoder dispatchThreads:env_threads threadsPerThreadgroup:env_group];
            [rollout_encoder endEncoding];

            id<MTLComputeCommandEncoder> reduce_encoder = [command computeCommandEncoder];
            [reduce_encoder setComputePipelineState:reduce];
            [reduce_encoder setBuffer:partials offset:0 atIndex:0];
            [reduce_encoder setBuffer:gradients offset:0 atIndex:1];
            [reduce_encoder setBuffer:params_buffer offset:0 atIndex:2];
            [reduce_encoder dispatchThreads:weight_threads threadsPerThreadgroup:weight_group];
            [reduce_encoder endEncoding];

            id<MTLComputeCommandEncoder> update_encoder = [command computeCommandEncoder];
            [update_encoder setComputePipelineState:update];
            [update_encoder setBuffer:weights offset:0 atIndex:0];
            [update_encoder setBuffer:gradients offset:0 atIndex:1];
            [update_encoder setBuffer:momentum offset:0 atIndex:2];
            [update_encoder setBuffer:variance offset:0 atIndex:3];
            [update_encoder setBuffer:params_buffer offset:0 atIndex:4];
            [update_encoder dispatchThreads:weight_threads threadsPerThreadgroup:weight_group];
            [update_encoder endEncoding];

            [command commit];
            last_command = command;
        }
        [last_command waitUntilCompleted];

        double elapsed = now_seconds() - start;
        double total_steps = (double)num_envs * (double)steps * (double)iterations;
        double steps_per_second = elapsed > 0.0 ? total_steps / elapsed : 0.0;
        float *w = (float *)weights.contents;
        printf("metal_es_train pendulums=%u features=%u weights=%u threadgroup=%u steps=%.0f elapsed=%.6fs sps=%.0f weights_checksum=%.6f first_weight=%.6f\n",
               num_pendulums, features, weights_total, threadgroup_size, total_steps, elapsed, steps_per_second, w[0], w[0]);
        if (output_path != NULL) {
            FILE *file = fopen(output_path, "wb");
            if (file == NULL) {
                fprintf(stderr, "could not open output path: %s\n", output_path);
                return 1;
            }
            size_t written = fwrite(w, sizeof(float), weights_total, file);
            fclose(file);
            if (written != weights_total) {
                fprintf(stderr, "could not write all weights\n");
                return 1;
            }
        }
    }
    return 0;
}
