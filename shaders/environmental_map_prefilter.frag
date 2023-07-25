#include "include/common.glsl"
#include "include/types.glsl"
#include "include/functions.glsl"

layout(push_constant, std430) uniform PushConstants {
    mat4 matrix;
    uint image_index;
    float roughness;
};

layout(location = 0) in vec3 in_local_pos;
layout(location = 0) out vec4 out_color;

const uint SAMPLE_COUNT = 1024u * 4;


vec3 prefilterEnvMap(vec3 R, float roughness)
{
	vec3 N = R;
	vec3 V = R;
	vec3 color = vec3(0.0);
	float totalWeight = 0.0;
	float envMapDim = float(textureSize(GetSampledTextureCube(image_index), 0).x);
	for(uint i = 0u; i < SAMPLE_COUNT; i++) {
		vec2 Xi = hammersley_2d(i, SAMPLE_COUNT);
		vec3 H = importance_sample_ggx(Xi, roughness, N);
		vec3 L = 2.0 * dot(V, H) * H - V;
		float dotNL = clamp(dot(N, L), 0.0, 1.0);
		if(dotNL > 0.0) {
			float dotNH = clamp(dot(N, H), 0.0, 1.0);
			float dotVH = clamp(dot(V, H), 0.0, 1.0);

			// Probability Distribution Function
			float pdf = distribution_ggx(dotNH, roughness) * dotNH / (4.0 * dotVH) + 0.0001;
			// Slid angle of current smple
			float omegaS = 1.0 / (float(SAMPLE_COUNT) * pdf);
			// Solid angle of 1 pixel across all cube faces
			float omegaP = 4.0 * PI / (6.0 * envMapDim * envMapDim);
			// Biased (+1.0) mip level for better result
			float mipLevel = roughness == 0.0 ? 0.0 : max(0.5 * log2(omegaS / omegaP) + 1.0, 0.0f);
			color += textureLod(GetSampledTextureCube(image_index), L, mipLevel).rgb * dotNL;
			totalWeight += dotNL;

		}
	}
	return (color / totalWeight);
}

void main() {
    out_color = vec4(prefilterEnvMap(normalize(in_local_pos), roughness), 1.0);
}