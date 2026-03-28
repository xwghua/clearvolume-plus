#version 330 core

// v_tex_coord is received from the vertex shader (kept for linking compatibility)
// but NDC is computed from gl_FragCoord for macOS robustness.
in vec2 v_tex_coord;
out vec4 frag_color;

// --- Volume and transfer function -----------------------------------------
uniform sampler3D volumeTex;    // 3-D volume, values in [0,1]
uniform sampler2D transferTex;  // 1-D LUT stored as single-row 2-D texture

// --- Camera / transform ---------------------------------------------------
// Direct ray construction: camera is fixed at (0, 0, cameraDistance) in world
// space; the volume is transformed via modelMatrix / inverseModelMatrix.
uniform float cameraDistance;   // distance from camera to world origin
uniform float tanHalfFov;       // tan(fovY / 2)
uniform float aspectRatio;      // viewport width / viewport height
// modelMatrix rotates/scales the unit AABB ([-0.5,0.5]^3) in world space.
uniform mat4 modelMatrix;
uniform mat4 inverseModelMatrix;

// --- Viewport size (physical pixels) for gl_FragCoord → NDC --------------
uniform float viewportW;
uniform float viewportH;

// --- Volume physical size -------------------------------------------------
// Aspect ratio of the volume in world space (max dimension = 1.0).
uniform vec3 volumeAspect;      // (rx, ry, rz), max component = 1.0

// --- Rendering parameters -------------------------------------------------
uniform float brightness;       // [0..5], default 1.0
uniform float gamma;            // [0.01..5], default 1.0
uniform float rangeMin;         // intensity normalisation lower bound [0,1]
uniform float rangeMax;         // intensity normalisation upper bound [0,1]
uniform float quality;          // [0.1..1], controls step count
uniform float dithering;        // [0..1], random ray-start offset strength

// --- Render mode ----------------------------------------------------------
uniform int renderMode;         // 0 = MaxProjection, 1 = IsoSurface

// --- Iso-surface parameters -----------------------------------------------
uniform float isoValue;         // threshold in [0,1] (normalised intensity)
uniform vec3  lightDir;         // world-space normalised light direction

// --- Dithering noise seed (changed each frame) ----------------------------
uniform float noiseSeed;        // random float [0,1] updated per frame

// --- ROI clipping (texture coordinate space [0,1]) -----------------------
uniform vec3 roiMin;            // lower bound per axis, default (0,0,0)
uniform vec3 roiMax;            // upper bound per axis, default (1,1,1)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pseudo-random hash for dithering (fast, good enough for anti-banding).
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7)) + noiseSeed) * 43758.5453);
}

// Ray-AABB intersection.  The box spans [-0.5*aspect, +0.5*aspect].
// Returns true if the ray hits; tMin/tMax are entry/exit distances.
bool intersectBox(vec3 rayOrigin, vec3 rayDir, vec3 boxHalf,
                  out float tMin, out float tMax) {
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (-boxHalf - rayOrigin) * invDir;
    vec3 t1 = ( boxHalf - rayOrigin) * invDir;
    vec3 tSmall = min(t0, t1);
    vec3 tLarge = max(t0, t1);
    tMin = max(max(tSmall.x, tSmall.y), tSmall.z);
    tMax = min(min(tLarge.x, tLarge.y), tLarge.z);
    return tMax > max(tMin, 0.0);
}

// Map a model-space position inside the AABB to texture coordinates [0,1]^3.
vec3 worldToTex(vec3 worldPos, vec3 boxHalf) {
    return worldPos / (2.0 * boxHalf) + 0.5;
}

// Normalise and gamma-correct a raw voxel intensity.
float mapIntensity(float raw) {
    float range = rangeMax - rangeMin;
    if (range < 1e-6) return 0.0;
    float v = clamp((raw - rangeMin) / range, 0.0, 1.0);
    return pow(v, gamma);
}

// Sample transfer function for a given mapped intensity.
// transferTex is a single-row 2D texture; sample along the X axis.
vec4 sampleTF(float mappedIntensity) {
    vec4 color = texture(transferTex, vec2(mappedIntensity, 0.5));
    color.rgb *= brightness;
    return color;
}

// Estimate volume gradient (central differences) for Phong shading.
vec3 estimateNormal(vec3 texCoord) {
    float dx = 1.0 / float(textureSize(volumeTex, 0).x);
    float dy = 1.0 / float(textureSize(volumeTex, 0).y);
    float dz = 1.0 / float(textureSize(volumeTex, 0).z);
    float gx = texture(volumeTex, texCoord + vec3(dx, 0, 0)).r
             - texture(volumeTex, texCoord - vec3(dx, 0, 0)).r;
    float gy = texture(volumeTex, texCoord + vec3(0, dy, 0)).r
             - texture(volumeTex, texCoord - vec3(0, dy, 0)).r;
    float gz = texture(volumeTex, texCoord + vec3(0, 0, dz)).r
             - texture(volumeTex, texCoord - vec3(0, 0, dz)).r;
    float len = length(vec3(gx, gy, gz));
    if (len < 1e-6) return vec3(0.0, 1.0, 0.0);  // fallback
    return normalize(vec3(gx, gy, gz));
}

// Phong shading — diffuse + specular, ambient = 0.1.
float phong(vec3 normal, vec3 rayDir) {
    vec3 L = normalize(lightDir);
    vec3 N = normalize(normal);
    // Ensure normal faces the viewer.
    if (dot(N, -rayDir) < 0.0) N = -N;
    float diff = max(dot(N, L), 0.0);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(-rayDir, R), 0.0), 10.0);
    return 0.1 + 0.6 * diff + 0.3 * spec;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
    // --- Reconstruct world-space ray (direct construction) ----------------
    // Use v_tex_coord for NDC — it interpolates linearly across the full-screen
    // quad (w=1 everywhere, no perspective correction needed) and is immune to
    // macOS Retina DPR scaling that affects gl_FragCoord window coordinates.
    vec2 ndc = v_tex_coord * 2.0 - 1.0;

    // Camera is fixed at (0, 0, cameraDistance) in world space.
    // Ray direction in world space: for NDC (nx, ny), the view-space direction is
    //   (nx * aspectRatio * tanHalfFov,  ny * tanHalfFov,  -1).
    // Our view matrix is a pure Z-translation (no rotation), so world == view.
    vec3 rayOrigin   = vec3(0.0, 0.0, cameraDistance);
    vec3 rayDirWorld = normalize(vec3(ndc.x * aspectRatio * tanHalfFov,
                                      ndc.y * tanHalfFov,
                                      -1.0));

    // --- Transform ray into model (volume) space --------------------------
    // The volume AABB is [-0.5*aspect, +0.5*aspect] in model space.
    vec3 rayOriginM = (inverseModelMatrix * vec4(rayOrigin,   1.0)).xyz;
    vec3 rayDirM    = normalize((inverseModelMatrix * vec4(rayDirWorld, 0.0)).xyz);
    vec3 boxHalf    = volumeAspect * 0.5;

    // --- Ray-AABB intersection --------------------------------------------
    float tMin, tMax;
    if (!intersectBox(rayOriginM, rayDirM, boxHalf, tMin, tMax)) {
        frag_color = vec4(0.0);
        return;
    }
    tMin = max(tMin, 0.0);

    // --- Adaptive step size -----------------------------------------------
    // Base: traverse the longest diagonal in ~300 * quality steps.
    float diagLen = length(2.0 * boxHalf);
    int   nSteps  = int(diagLen / (0.003 / max(quality, 0.01)));
    nSteps = clamp(nSteps, 50, 800);
    float stepSize = (tMax - tMin) / float(nSteps);

    // --- Optional dithering offset ----------------------------------------
    float jitter = dithering > 0.0
        ? hash(v_tex_coord) * stepSize * dithering
        : 0.0;
    float tStart = tMin + jitter;

    // -----------------------------------------------------------------------
    // Render mode dispatch
    // -----------------------------------------------------------------------

    if (renderMode == 0) {
        // -------------------------------------------------------------------
        // Maximum Intensity Projection (MIP)
        // -------------------------------------------------------------------
        vec4  maxColor  = vec4(0.0);
        float maxMapped = -1.0;

        for (int i = 0; i < nSteps; ++i) {
            float t = tStart + float(i) * stepSize;
            if (t > tMax) break;

            vec3 pos    = rayOriginM + t * rayDirM;
            vec3 texPos = worldToTex(pos, boxHalf);

            if (any(lessThan(texPos, vec3(0.001))) ||
                any(greaterThan(texPos, vec3(0.999)))) continue;

            if (any(lessThan(texPos, roiMin)) ||
                any(greaterThan(texPos, roiMax))) continue;

            float raw    = texture(volumeTex, texPos).r;
            float mapped = mapIntensity(raw);
            if (mapped > maxMapped) {
                maxMapped = mapped;
                maxColor  = sampleTF(mapped);
            }
        }

        frag_color = maxColor;

    } else {
        // -------------------------------------------------------------------
        // Iso-surface rendering
        // -------------------------------------------------------------------
        float prevVal  = 0.0;
        bool  hit      = false;
        vec4  isoColor = vec4(0.0);

        for (int i = 0; i < nSteps; ++i) {
            float t = tStart + float(i) * stepSize;
            if (t > tMax) break;

            vec3 pos    = rayOriginM + t * rayDirM;
            vec3 texPos = worldToTex(pos, boxHalf);

            if (any(lessThan(texPos, vec3(0.001))) ||
                any(greaterThan(texPos, vec3(0.999)))) {
                prevVal = 0.0;
                continue;
            }

            if (any(lessThan(texPos, roiMin)) ||
                any(greaterThan(texPos, roiMax))) {
                prevVal = 0.0;
                continue;
            }

            float raw    = texture(volumeTex, texPos).r;
            float mapped = mapIntensity(raw);

            if (prevVal < isoValue && mapped >= isoValue) {
                float alpha  = (isoValue - prevVal) / (mapped - prevVal + 1e-6);
                float tHit   = t - stepSize * (1.0 - alpha);
                vec3  posHit = rayOriginM + tHit * rayDirM;
                vec3  texHit = worldToTex(posHit, boxHalf);

                vec3  N     = estimateNormal(texHit);
                float shade = phong(N, rayDirM);
                isoColor     = sampleTF(isoValue);
                isoColor.rgb *= shade;
                isoColor.a = 1.0;
                hit = true;
                break;
            }
            prevVal = mapped;
        }

        frag_color = hit ? isoColor : vec4(0.0);
    }
}
