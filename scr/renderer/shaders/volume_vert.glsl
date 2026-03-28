#version 330 core

// Full-screen quad vertices are passed in clip space (-1..1).
// tex_coord is passed through to the fragment shader.
//
// Explicit layout locations are REQUIRED for cross-platform compatibility.
// Without them, Apple's macOS OpenGL driver may assign locations in a
// different order than Windows/Linux drivers, causing position and tex_coord
// to be swapped (misaligned volume rendering).

layout(location = 0) in vec2 position;    // clip-space XY of the quad vertex
layout(location = 1) in vec2 tex_coord;   // [0,1] texture coordinates

out vec2 v_tex_coord;

void main() {
    v_tex_coord = tex_coord;
    gl_Position = vec4(position, 0.0, 1.0);
}
