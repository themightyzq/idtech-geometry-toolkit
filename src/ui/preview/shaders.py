"""
GLSL shaders for the preview widget.

Uses simple Phong shading with per-texture colors derived from texture name hashes.
"""

# Vertex shader - transforms vertices and passes normals to fragment shader
VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec3 aColor;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 VertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    VertexColor = aColor;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

# Fragment shader - Phong lighting with ambient, diffuse, and specular
FRAGMENT_SHADER = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 VertexColor;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform float ambientStrength;
uniform float specularStrength;
uniform sampler2D textureSampler;
uniform bool useTexture;

void main() {
    // Get base color from texture or vertex color
    vec3 baseColor;
    if (useTexture) {
        baseColor = texture(textureSampler, TexCoord).rgb;
    } else {
        baseColor = VertexColor;
    }

    // Ambient
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * baseColor;
    FragColor = vec4(result, 1.0);
}
"""

# Simpler shaders for OpenGL 2.1 fallback (macOS compatibility)
# Note: Legacy mode uses fixed-function pipeline, not shaders.
# These are kept for reference but _render_solid_fixed_function handles rendering.
VERTEX_SHADER_LEGACY = """
#version 120

attribute vec3 aPos;
attribute vec3 aNormal;
attribute vec2 aTexCoord;
attribute vec3 aColor;

varying vec3 FragPos;
varying vec3 Normal;
varying vec2 TexCoord;
varying vec3 VertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = (model * vec4(aPos, 1.0)).xyz;
    Normal = aNormal;
    TexCoord = aTexCoord;
    VertexColor = aColor;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER_LEGACY = """
#version 120

varying vec3 FragPos;
varying vec3 Normal;
varying vec2 TexCoord;
varying vec3 VertexColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform float ambientStrength;
uniform float specularStrength;
uniform sampler2D textureSampler;
uniform bool useTexture;

void main() {
    // Get base color from texture or vertex color
    vec3 baseColor;
    if (useTexture) {
        baseColor = texture2D(textureSampler, TexCoord).rgb;
    } else {
        baseColor = VertexColor;
    }

    // Ambient
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * baseColor;
    gl_FragColor = vec4(result, 1.0);
}
"""

# Wireframe shader - simple solid color pass-through
WIREFRAME_VERTEX = """
#version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

WIREFRAME_FRAGMENT = """
#version 330 core

out vec4 FragColor;
uniform vec3 wireColor;

void main() {
    FragColor = vec4(wireColor, 1.0);
}
"""

WIREFRAME_VERTEX_LEGACY = """
#version 120

attribute vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

WIREFRAME_FRAGMENT_LEGACY = """
#version 120

uniform vec3 wireColor;

void main() {
    gl_FragColor = vec4(wireColor, 1.0);
}
"""
