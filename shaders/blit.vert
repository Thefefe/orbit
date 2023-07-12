#include "include/common.glsl"
#include "include/types.glsl"

const vec2 VERTICES[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0)
);

layout(location = 0) out vec2 uv;

void main() {
    uv = VERTICES[gl_VertexIndex];
    uv.y = 1.0 - uv.y;
    gl_Position = vec4(VERTICES[gl_VertexIndex] * 2.0 - 1.0, 0.0, 1.0);
}