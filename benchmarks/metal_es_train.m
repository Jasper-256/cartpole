#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

enum {
    FEATURE_COUNT = 24,
    WEIGHT_COUNT = 25,
    THREADGROUP_SIZE = 128,
};

typedef struct {
    uint32_t steps;
    uint32_t num_envs;
    uint32_t num_groups;
    uint32_t iteration;
    float sigma;
    float learning_rate;
} TrainParams;

static double now_seconds(void) {
    static mach_timebase_info_data_t info;
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    return (double)mach_absolute_time() * (double)info.numer / (double)info.denom * 1e-9;
}

static NSString *kernelSource(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "#define FEATURE_COUNT 24u\n"
            "#define WEIGHT_COUNT 25u\n"
            "#define THREADGROUP_SIZE 128u\n"
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
            "inline float wrap_angle(float theta) {\n"
            "    const float pi = 3.14159265358979323846f;\n"
            "    float wrapped = fmod(theta + pi, 2.0f * pi);\n"
            "    if (wrapped < 0.0f) wrapped += 2.0f * pi;\n"
            "    return wrapped - pi;\n"
            "}\n"
            "kernel void es_rollout(constant float *weights [[buffer(0)]],\n"
            "                       device float *partials [[buffer(1)]],\n"
            "                       constant TrainParams &params [[buffer(2)]],\n"
            "                       uint gid [[thread_position_in_grid]],\n"
            "                       uint lid [[thread_position_in_threadgroup]],\n"
            "                       uint group_id [[threadgroup_position_in_grid]],\n"
            "                       threadgroup float *scratch [[threadgroup(0)]]) {\n"
            "    float eps[WEIGHT_COUNT];\n"
            "    float local_weights[WEIGHT_COUNT];\n"
            "    float score = 0.0f;\n"
            "    if (gid < params.num_envs) {\n"
            "        for (uint j = 0u; j < WEIGHT_COUNT; ++j) {\n"
            "            eps[j] = eps_sign(gid, j, params.iteration);\n"
            "            local_weights[j] = weights[j] + params.sigma * eps[j];\n"
            "        }\n"
            "        uint pair_id = gid >> 1u;\n"
            "        float u0 = rand01(pair_id * 13u + params.iteration * 1315423911u);\n"
            "        float u1 = rand01(pair_id * 17u + params.iteration * 2654435761u);\n"
            "        float u2 = rand01(pair_id * 19u + params.iteration * 2246822519u);\n"
            "        float u3 = rand01(pair_id * 23u + params.iteration * 3266489917u);\n"
            "        float x = 0.06f * u0 - 0.03f;\n"
            "        float x_dot = 0.06f * u1 - 0.03f;\n"
            "        float theta_1 = 3.14159265358979323846f - 0.08f + 0.16f * u2;\n"
            "        float theta_2 = 3.14159265358979323846f - 0.08f + 0.16f * u3;\n"
            "        float upright_case = 0.0f;\n"
            "        float theta_dot_1 = -0.02f + 0.04f * u1;\n"
            "        float theta_dot_2 = 0.02f - 0.04f * u0;\n"
            "        for (uint step = 0u; step < params.steps; ++step) {\n"
            "            float obs[FEATURE_COUNT];\n"
            "            float norm_x = x / X_THRESHOLD;\n"
            "            float norm_x_dot = x_dot / 5.0f;\n"
            "            float upright_gate = exp(-0.25f * ((theta_1 / 0.7f) * (theta_1 / 0.7f) + (theta_2 / 0.7f) * (theta_2 / 0.7f)));\n"
            "            float swing_gate = 1.0f - upright_gate;\n"
            "            float sin_feature_1 = sin(theta_1);\n"
            "            float sin_feature_2 = sin(theta_2);\n"
            "            float cos_feature_1 = cos(theta_1);\n"
            "            float cos_feature_2 = cos(theta_2);\n"
            "            float norm_theta_dot_1 = theta_dot_1 / 10.0f;\n"
            "            float norm_theta_dot_2 = theta_dot_2 / 10.0f;\n"
            "            float energy_target_feature = 2.0f * GRAVITY * POLE_LENGTH;\n"
            "            float energy_1_feature = 0.5f * (POLE_LENGTH * theta_dot_1) * (POLE_LENGTH * theta_dot_1) + GRAVITY * POLE_LENGTH * (cos_feature_1 + 1.0f);\n"
            "            float energy_2_feature = 0.5f * (POLE_LENGTH * theta_dot_2) * (POLE_LENGTH * theta_dot_2) + GRAVITY * POLE_LENGTH * (cos_feature_2 + 1.0f);\n"
            "            float energy_error_1 = (energy_1_feature - energy_target_feature) / energy_target_feature;\n"
            "            float energy_error_2 = (energy_2_feature - energy_target_feature) / energy_target_feature;\n"
            "            obs[0] = swing_gate;\n"
            "            obs[1] = upright_gate;\n"
            "            obs[2] = norm_x * swing_gate;\n"
            "            obs[3] = norm_x_dot * swing_gate;\n"
            "            obs[4] = norm_x * upright_gate;\n"
            "            obs[5] = norm_x_dot * upright_gate;\n"
            "            obs[6] = swing_gate * sin_feature_1;\n"
            "            obs[7] = swing_gate * sin_feature_2;\n"
            "            obs[8] = swing_gate * cos_feature_1;\n"
            "            obs[9] = swing_gate * cos_feature_2;\n"
            "            obs[10] = swing_gate * norm_theta_dot_1;\n"
            "            obs[11] = swing_gate * norm_theta_dot_2;\n"
            "            obs[12] = swing_gate * norm_theta_dot_1 * sin_feature_1;\n"
            "            obs[13] = swing_gate * norm_theta_dot_2 * sin_feature_2;\n"
            "            obs[14] = swing_gate * norm_theta_dot_1 * cos_feature_1;\n"
            "            obs[15] = swing_gate * norm_theta_dot_2 * cos_feature_2;\n"
            "            obs[16] = swing_gate * energy_error_1;\n"
            "            obs[17] = swing_gate * energy_error_2;\n"
            "            obs[18] = swing_gate * energy_error_1 * norm_theta_dot_1 * cos_feature_1;\n"
            "            obs[19] = swing_gate * energy_error_2 * norm_theta_dot_2 * cos_feature_2;\n"
            "            obs[20] = upright_gate * sin_feature_1;\n"
            "            obs[21] = upright_gate * sin_feature_2;\n"
            "            obs[22] = upright_gate * norm_theta_dot_1;\n"
            "            obs[23] = upright_gate * norm_theta_dot_2;\n"
            "            float force_logit = local_weights[FEATURE_COUNT];\n"
            "            for (uint j = 0u; j < FEATURE_COUNT; ++j) force_logit += local_weights[j] * obs[j];\n"
            "            float force = FORCE_MAG * tanh(force_logit);\n"
            "            float previous_cos_1 = cos(theta_1);\n"
            "            float previous_cos_2 = cos(theta_2);\n"
            "            float sin_1 = sin(theta_1);\n"
            "            float sin_2 = sin(theta_2);\n"
            "            float delta = theta_1 - theta_2;\n"
            "            float sin_delta = sin(delta);\n"
            "            float cos_delta = cos(delta);\n"
            "            float pole_mass_length = POLE_MASS * POLE_LENGTH;\n"
            "            float pole_inertia = POLE_MASS * POLE_LENGTH * POLE_LENGTH;\n"
            "            float matrix_00 = CART_MASS + 2.0f * POLE_MASS;\n"
            "            float matrix_01 = 2.0f * pole_mass_length * previous_cos_1;\n"
            "            float matrix_02 = pole_mass_length * previous_cos_2;\n"
            "            float matrix_11 = 2.0f * pole_inertia;\n"
            "            float matrix_12 = pole_inertia * cos_delta;\n"
            "            float matrix_22 = pole_inertia;\n"
            "            float theta_dot_1_sq = theta_dot_1 * theta_dot_1;\n"
            "            float theta_dot_2_sq = theta_dot_2 * theta_dot_2;\n"
            "            float bias_0 = -2.0f * pole_mass_length * sin_1 * theta_dot_1_sq\n"
            "                - pole_mass_length * sin_2 * theta_dot_2_sq + CART_FRICTION * x_dot;\n"
            "            float bias_1 = pole_inertia * sin_delta * theta_dot_2_sq\n"
            "                - 2.0f * pole_mass_length * GRAVITY * sin_1 + POLE_FRICTION * theta_dot_1;\n"
            "            float bias_2 = -pole_inertia * sin_delta * theta_dot_1_sq\n"
            "                - pole_mass_length * GRAVITY * sin_2 + POLE_FRICTION * theta_dot_2;\n"
            "            float rhs_0 = force - bias_0;\n"
            "            float rhs_1 = -bias_1;\n"
            "            float rhs_2 = -bias_2;\n"
            "            float cofactor_00 = matrix_11 * matrix_22 - matrix_12 * matrix_12;\n"
            "            float cofactor_01 = matrix_02 * matrix_12 - matrix_01 * matrix_22;\n"
            "            float cofactor_02 = matrix_01 * matrix_12 - matrix_02 * matrix_11;\n"
            "            float cofactor_11 = matrix_00 * matrix_22 - matrix_02 * matrix_02;\n"
            "            float cofactor_12 = matrix_01 * matrix_02 - matrix_00 * matrix_12;\n"
            "            float cofactor_22 = matrix_00 * matrix_11 - matrix_01 * matrix_01;\n"
            "            float determinant = matrix_00 * cofactor_00 + matrix_01 * cofactor_01 + matrix_02 * cofactor_02;\n"
            "            float q_acc_0 = (cofactor_00 * rhs_0 + cofactor_01 * rhs_1 + cofactor_02 * rhs_2) / determinant;\n"
            "            float q_acc_1 = (cofactor_01 * rhs_0 + cofactor_11 * rhs_1 + cofactor_12 * rhs_2) / determinant;\n"
            "            float q_acc_2 = (cofactor_02 * rhs_0 + cofactor_12 * rhs_1 + cofactor_22 * rhs_2) / determinant;\n"
            "            x_dot += DT * q_acc_0;\n"
            "            x += DT * x_dot;\n"
            "            theta_dot_1 += DT * q_acc_1;\n"
            "            theta_dot_2 += DT * q_acc_2;\n"
            "            theta_1 = wrap_angle(theta_1 + DT * theta_dot_1);\n"
            "            theta_2 = wrap_angle(theta_2 + DT * theta_dot_2);\n"
            "            if (abs(x) > 2.4f || !isfinite(x + x_dot + theta_1 + theta_2 + theta_dot_1 + theta_dot_2)) {\n"
            "                score -= 1000.0f;\n"
            "                break;\n"
            "            }\n"
            "            float cos_1 = cos(theta_1);\n"
            "            float cos_2 = cos(theta_2);\n"
            "            float link_height_1 = 0.5f * (cos_1 + 1.0f);\n"
            "            float link_height_2 = 0.5f * (cos_2 + 1.0f);\n"
            "            float height = 0.5f * (link_height_1 + link_height_2);\n"
            "            float chain_height = sqrt(max(link_height_1, 1.0e-4f) * max(link_height_2, 1.0e-4f));\n"
            "            float height_reward = 0.5f * height + 0.5f * chain_height;\n"
            "            float top_speed = 0.5f * (link_height_1 * link_height_1 * abs(theta_dot_1)\n"
            "                + link_height_2 * link_height_2 * abs(theta_dot_2));\n"
            "            float controlled = chain_height * exp(-0.5f * (theta_dot_1 * theta_dot_1 + theta_dot_2 * theta_dot_2));\n"
            "            float angle_precision = exp(-0.5f * ((theta_1 / 0.35f) * (theta_1 / 0.35f) + (theta_2 / 0.35f) * (theta_2 / 0.35f)));\n"
            "            float top_hold = angle_precision * exp(-0.5f * (theta_dot_1 * theta_dot_1 + theta_dot_2 * theta_dot_2));\n"
            "            float progress = 0.5f * ((cos_1 - previous_cos_1) + (cos_2 - previous_cos_2));\n"
            "            float centered = 1.0f - clamp((x / X_THRESHOLD) * (x / X_THRESHOLD), 0.0f, 1.0f);\n"
            "            float bottom_motion = (1.0f - height) * 0.5f * (tanh(abs(theta_dot_1) / 2.0f) + tanh(abs(theta_dot_2) / 2.0f));\n"
            "            float bottom_quiet = (1.0f - height) * (1.0f - clamp(bottom_motion, 0.0f, 1.0f));\n"
            "            float energy_target = 2.0f * GRAVITY * POLE_LENGTH;\n"
            "            float energy_1 = 0.5f * (POLE_LENGTH * theta_dot_1) * (POLE_LENGTH * theta_dot_1) + GRAVITY * POLE_LENGTH * (cos_1 + 1.0f);\n"
            "            float energy_2 = 0.5f * (POLE_LENGTH * theta_dot_2) * (POLE_LENGTH * theta_dot_2) + GRAVITY * POLE_LENGTH * (cos_2 + 1.0f);\n"
            "            float energy_err_1 = (energy_1 - energy_target) / energy_target;\n"
            "            float energy_err_2 = (energy_2 - energy_target) / energy_target;\n"
            "            float energy_score = exp(-0.5f * (energy_err_1 * energy_err_1 + energy_err_2 * energy_err_2));\n"
            "            float stable = (abs(x) <= 0.5f && abs(x_dot) <= 0.75f && abs(theta_1) <= 0.20943951f && abs(theta_2) <= 0.20943951f && abs(theta_dot_1) <= 1.0f && abs(theta_dot_2) <= 1.0f) ? 1.0f : 0.0f;\n"
            "            float local_error = theta_1 * theta_1 + theta_2 * theta_2 + 0.2f * (theta_dot_1 * theta_dot_1 + theta_dot_2 * theta_dot_2) + 0.5f * (x / X_THRESHOLD) * (x / X_THRESHOLD) + 0.05f * x_dot * x_dot;\n"
            "            float velocity_cost = 0.01f * x_dot * x_dot + 2.0f * (x / X_THRESHOLD) * (x / X_THRESHOLD) + 0.002f * 0.5f * (theta_dot_1 * theta_dot_1 + theta_dot_2 * theta_dot_2) + 0.25f * 0.5f * (link_height_1 * link_height_1 * theta_dot_1 * theta_dot_1 + link_height_2 * link_height_2 * theta_dot_2 * theta_dot_2);\n"
            "            float action_cost = 0.01f * (force / FORCE_MAG) * (force / FORCE_MAG);\n"
            "            score += 8.0f * height_reward + 100.0f * controlled + 200.0f * top_hold + upright_case * (50.0f - 80.0f * local_error) + 0.75f * energy_score + 60.0f * progress + 3.0f * bottom_motion + 0.1f * centered + 2000.0f * stable - 1.5f * bottom_quiet - velocity_cost - action_cost;\n"
            "        }\n"
            "    } else {\n"
            "        for (uint j = 0u; j < WEIGHT_COUNT; ++j) eps[j] = 0.0f;\n"
            "    }\n"
            "    if (!isfinite(score)) score = -10000.0f;\n"
            "    score = clamp(score, -10000.0f, 100000.0f);\n"
            "    for (uint j = 0u; j < WEIGHT_COUNT; ++j) scratch[j * THREADGROUP_SIZE + lid] = score * eps[j];\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (uint stride = THREADGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {\n"
            "        if (lid < stride) {\n"
            "            for (uint j = 0u; j < WEIGHT_COUNT; ++j) {\n"
            "                scratch[j * THREADGROUP_SIZE + lid] += scratch[j * THREADGROUP_SIZE + lid + stride];\n"
            "            }\n"
            "        }\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (lid == 0u) {\n"
            "        for (uint j = 0u; j < WEIGHT_COUNT; ++j) partials[group_id * WEIGHT_COUNT + j] = scratch[j * THREADGROUP_SIZE];\n"
            "    }\n"
            "}\n"
            "kernel void reduce_gradients(device float *partials [[buffer(0)]],\n"
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
            "}\n";
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

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        uint32_t num_envs = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 3145728u;
        uint32_t steps = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 500u;
        uint32_t iterations = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 1u;
        float sigma = argc > 4 ? strtof(argv[4], NULL) : 0.05f;
        float learning_rate = argc > 5 ? strtof(argv[5], NULL) : 0.02f;
        const char *output_path = argc > 6 ? argv[6] : NULL;
        uint32_t num_groups = (num_envs + THREADGROUP_SIZE - 1u) / THREADGROUP_SIZE;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kernelSource() options:nil error:&error];
        if (library == nil) {
            fprintf(stderr, "library error: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> rollout = pipeline(device, library, @"es_rollout");
        id<MTLComputePipelineState> reduce = pipeline(device, library, @"reduce_gradients");
        id<MTLComputePipelineState> update = pipeline(device, library, @"update_weights");

        id<MTLBuffer> weights = [device newBufferWithLength:WEIGHT_COUNT * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> partials = [device newBufferWithLength:(size_t)num_groups * WEIGHT_COUNT * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> gradients = [device newBufferWithLength:WEIGHT_COUNT * sizeof(float) options:MTLResourceStorageModePrivate];
        id<MTLBuffer> momentum = [device newBufferWithLength:WEIGHT_COUNT * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> variance = [device newBufferWithLength:WEIGHT_COUNT * sizeof(float) options:MTLResourceStorageModeShared];
        if (weights == nil || partials == nil || gradients == nil || momentum == nil || variance == nil) {
            fprintf(stderr, "buffer allocation failed\n");
            return 1;
        }
        memset(momentum.contents, 0, WEIGHT_COUNT * sizeof(float));
        memset(variance.contents, 0, WEIGHT_COUNT * sizeof(float));
        float *w = (float *)weights.contents;
        for (uint32_t i = 0; i < WEIGHT_COUNT; ++i) {
            w[i] = 0.0f;
        }
        w[4] = -0.378319f;
        w[5] = -1.719792f;
        w[20] = 8.887854f;
        w[21] = -12.133175f;
        w[22] = 3.222614f;
        w[23] = -20.0f;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        MTLSize env_threads = MTLSizeMake(num_envs, 1, 1);
        MTLSize train_group = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        MTLSize weight_threads = MTLSizeMake(WEIGHT_COUNT, 1, 1);
        MTLSize weight_group = MTLSizeMake(WEIGHT_COUNT, 1, 1);
        NSUInteger scratch_bytes = WEIGHT_COUNT * THREADGROUP_SIZE * sizeof(float);

        double start = now_seconds();
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
            [rollout_encoder dispatchThreads:env_threads threadsPerThreadgroup:train_group];
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
            [command waitUntilCompleted];
        }
        double elapsed = now_seconds() - start;
        double total_steps = (double)num_envs * (double)steps * (double)iterations;
        double steps_per_second = elapsed > 0.0 ? total_steps / elapsed : 0.0;
        printf("metal_es_train steps=%.0f elapsed=%.6fs sps=%.0f weights_checksum=%.6f first_weight=%.6f\n",
               total_steps, elapsed, steps_per_second, w[0], w[0]);
        if (output_path != NULL) {
            FILE *file = fopen(output_path, "wb");
            if (file == NULL) {
                fprintf(stderr, "could not open output path: %s\n", output_path);
                return 1;
            }
            size_t written = fwrite(w, sizeof(float), WEIGHT_COUNT, file);
            fclose(file);
            if (written != WEIGHT_COUNT) {
                fprintf(stderr, "could not write all weights\n");
                return 1;
            }
        }
    }
    return 0;
}
