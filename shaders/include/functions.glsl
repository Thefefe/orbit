// Based omn http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
float random(vec2 co)
{
	float a = 12.9898;
	float b = 78.233;
	float c = 43758.5453;
	float dt= dot(co.xy ,vec2(a,b));
	float sn= mod(dt,3.14);
	return fract(sin(sn) * c);
}

uint div_ceil(uint lhs, uint rhs) {
    uint d = lhs / rhs;
    uint r = lhs % rhs;
    if (r > 0 && rhs > 0) {
        return d + 1;
    } else {
        return d;
    }
}

uint hash(uint a)
{
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);
   return a;
}

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec2 hammersley_2d(uint i, uint N) 
{
	// Radical inverse based on http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
	uint bits = (i << 16u) | (i >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	float rdi = float(bits) * 2.3283064365386963e-10;
	return vec2(float(i) /float(N), rdi);
}

// Based on http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_slides.pdf
vec3 importance_sample_ggx(vec2 Xi, float roughness, vec3 normal) 
{
	// Maps a 2D point to a hemisphere with spread based on roughness
	float alpha = roughness * roughness;
	float phi = 2.0 * PI * Xi.x + random(normal.xz) * 0.1;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (alpha*alpha - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

	// Tangent space
	vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangentX = normalize(cross(up, normal));
	vec3 tangentY = normalize(cross(normal, tangentX));

	// Convert to world Space
	return normalize(tangentX * H.x + tangentY * H.y + normal * H.z);
}

float distribution_ggx(float n_dot_h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return a2 / max(denom, EPSILON);
}

float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k); // schlick ggx
    float ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float h_dot_v, vec3 base_reflectivity) {
    return base_reflectivity + (1.0 - base_reflectivity) * pow(1.0 - h_dot_v, 5.0);
}

vec3 fresnel_schlick_roughness(float cos_theta, vec3 base_reflectivity, float roughness)
{
    return base_reflectivity + (max(vec3(1.0 - roughness), base_reflectivity) - base_reflectivity)
     * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
vec3 reference_orthonormal_vector(vec3 v) {
    float signum = v.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (signum + v.z);
    float b = v.x * v.y * a;
    return vec3(b, signum + v.y * v.y * a, -v.y);
}

vec3 octahedron_decode(vec2 f) {
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z , 0.0);
    n.xy += mix(vec2(t), vec2(-t), greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}

void unpack_normal_tangent(i8vec4 packed, out vec3 n, out vec4 t) {
    vec4 unpacked = vec4(packed) / 127.0;
    
    n = octahedron_decode(unpacked.xy);

    vec3 reference_tangent = reference_orthonormal_vector(n);
    float tangent_alpha = unpacked.z * PI;
    
    vec3 tangent = reference_tangent * cos(tangent_alpha) + cross(reference_tangent, n) * sin(tangent_alpha);
    t = vec4(normalize(tangent), unpacked.w);
}

//https://github.com/kbinani/colormap-shaders/blob/master/shaders/glsl/MATLAB_jet.frag
float colormap_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}

float colormap_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}

float colormap_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}