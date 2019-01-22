#include "app.h"

#include <string>
#include <vector>
#include <fstream>
#include <stdint.h>
#include <sstream>
#include <cstdlib>
#include <stdint.h>
#include <algorithm>
#include <memory.h>

#include <IL/il.h>
#include <OptiXMesh.h>

using namespace std;
using namespace optix;

#define DEG2RAD(x) (x * 0.01745329251994329575)
#define RAD2DEG(x) (x * 57.2957795131)

const char* const SAMPLE_NAME = "caveoptix";
const int NUMBER_OF_BRDF_INDICES = 3;
const int NUMBER_OF_LIGHT_INDICES = 2;

static float rand_range(float min, float max)
{
    static unsigned int seed = 0u;
    return min + (max - min) * rnd(seed);
}

static const char *g_screenquad_vert =
	"#version 440 core\n"
	"layout(location = 0) in vec3 vertex;\n"
	"layout(location = 1) in vec3 normal;\n"
	"layout(location = 2) in vec3 texcoord;\n"
	"uniform vec4 uCoords;\n"
	"uniform vec2 uScreen;\n"
	"out vec3 vtc;\n"
	"void main() {\n"
	"   vtc = texcoord*0.5+0.5;\n"
	"   gl_Position = vec4( -1.0 + (uCoords.x/uScreen.x) + (vertex.x+1.0f)*(uCoords.z-uCoords.x)/uScreen.x,\n"
	"                       -1.0 + (uCoords.y/uScreen.y) + (vertex.y+1.0f)*(uCoords.w-uCoords.y)/uScreen.y,\n"
	"                       0.0f, 1.0f );\n"
	"}\n";

static const char *g_screenquad_frag =
	"#version 440\n"
	"uniform sampler2D uTex1;\n"
	"uniform sampler2D uTex2;\n"
	"uniform int uTexFlags;\n"
	"in vec3 vtc;\n"
	"out vec4 outColor;\n"
	"void main() {\n"
	"   vec4 op1 = ((uTexFlags & 0x01)==0) ? texture ( uTex1, vtc.xy) : texture ( uTex1, vec2(vtc.x, 1.0-vtc.y));\n"
	"   if ( (uTexFlags & 0x02) != 0 ) {\n"
	"		vec4 op2 = ((uTexFlags & 0x04)==0) ? texture ( uTex2, vtc.xy) : texture ( uTex2, vec2(vtc.x, 1.0-vtc.y));\n"
	"		outColor = vec4( op1.xyz*(1.0-op2.w) + op2.xyz * op2.w, 1 );\n"
	"   } else { \n"
	"		outColor = vec4( op1.xyz, 1 );\n"
	"   }\n"
	"}\n";


OptixApp::OptixApp(string sf): m_initialized(false), m_framecount(0),
						context(0), use_pbo(true), gl_screen_tex(0),
                        elapsedTime(0), lastTime(0),
                        accumulation_frame(0), transmittance_log_scale(0),
                        max_depth(3)
{
    scene_file = sf;
}

OptixApp::~OptixApp() {
	if(context) {
		context->destroy();
		context = 0;
	}
}

void OptixApp::init(int w, int h) {

    m_width = w;
    m_height = h;
    cout << "width: " << m_width << " height: " << m_height << endl;

    // GL
    initGL();

	// setup OPTIX
	if (scene_file == "") {
        scene_file = sutil::samplesDir() + std::string("/data/spaceship.scene");
        scene = LoadScene(scene_file.c_str(), m_width, m_height);
	}
	else {
		scene = LoadScene(scene_file.c_str(), m_width, m_height);
	}
    
    ilInit();

    createContext();

    // Load textures
    for (int i = 0; i < scene->texture_map.size(); i++) {
        Texture tex;
        Picture* picture = new Picture;
        std::string textureFilename = std::string(sutil::samplesDir()) + "/data/" + scene->texture_map[i];
        std::cout << textureFilename << std::endl;
        picture->load(textureFilename);
        tex.createSampler(context, picture);
        scene->textures.push_back(tex);
        delete picture;
    }

    // Set textures to albedo ID of materials
    for (int i = 0; i < scene->materials.size(); i++) {
        if(scene->materials[i].albedoID != RT_TEXTURE_ID_NULL) {
            scene->materials[i].albedoID = scene->textures[scene->materials[i].albedoID-1].getId();
        }
    }

    m_bufferLightParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_bufferLightParameters->setElementSize(sizeof(LightParameter));
    m_bufferLightParameters->setSize(scene->lights.size());
    updateLightParameters(scene->lights);
    context["sysLightParameters"]->setBuffer(m_bufferLightParameters);
    
    m_bufferMaterialParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_bufferMaterialParameters->setElementSize(sizeof(MaterialParameter));
    m_bufferMaterialParameters->setSize(scene->materials.size());
    updateMaterialParameters(scene->materials);
    context["sysMaterialParameters"]->setBuffer(m_bufferMaterialParameters);

    context["sysNumberOfLights"]->setInt(scene->lights.size());
    optix::Group top_group;
    const optix::Aabb aabb = createGeometry(top_group);

    context->validate();

    camera_eye = optix::make_float3( 0.0f, 1.5f*aabb.extent( 1 ), 1.5f*aabb.extent( 2 ) );
    camera_lookat = aabb.center();
    camera_up = optix::make_float3( 0.0f, 1.0f, 0.0f );
    
	m_initialized = true;
}

// Converts the buffer format to gl format
GLenum glFormatFromBufferFormat(bufferPixelFormat pixel_format, RTformat buffer_format)
{
    if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
    {
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_BGRA;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGRA;
        default:
            cout << "Unknown buffer pixel format" << endl;
            exit(1);
            //throw Exception("Unknown buffer pixel format");
        }
    }
    else if (buffer_format == RT_FORMAT_FLOAT4)
    {
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGRA;
        default:
            cout << "Unknown buffer pixel format" << endl;
            exit(1);
            //throw Exception("Unknown buffer pixel format");
        }
    }
    else if (buffer_format == RT_FORMAT_FLOAT3)
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_RGB;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGB;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGR;
        default:
            cout << "Unknown buffer pixel format" << endl;
            exit(1);
            //throw Exception("Unknown buffer pixel format");
        }
    else if (buffer_format == RT_FORMAT_FLOAT)
        return GL_LUMINANCE;
    else {
        cout << "Unknown buffer pixel format" << endl;
        exit(1);
        //throw Exception("Unknown buffer format");
    }
}

void OptixApp::display(const float V[16], const float P[16], const float campos[3]) {
	if(!m_initialized) return;

    // update camera
    const float vfov = 45.0f;
    const float aspect_ratio = static_cast<float>(m_width) /
                               static_cast<float>(m_height);

    float3 camera_u, camera_v, camera_w;
    int TYPE = 0; // 0: calculate (U, V, W) from (MV, P, campos)

    if(TYPE == 0) {
        float b = P[5];
        float FOV = 2.0f * (float)atan(1.0f/b);
        float focal = 1 / tan(FOV/2);
        
        camera_eye = optix::make_float3(campos[0], campos[1], campos[2]);
        //camera_eye = optix::make_float3(V[12], V[13], V[14]);
        camera_u = optix::make_float3(V[0], V[4], V[8]);
        camera_v = -1*optix::make_float3(V[1], V[5], V[9]);
        camera_w = -1*optix::make_float3(V[2], V[6], V[10]);
        
        float fovY = 0.5 * FOV;
        float fovX = atan(tan(FOV)*aspect_ratio);
        float ulen = focal * tan(FOV); // * aspect_ratio;
        float vlen = focal * tan(fovY);
        camera_u = ulen * camera_u;
        camera_v = vlen * camera_v;
        camera_w = focal * camera_w;
        
        if(m_framecount < 1) {
            cout << "FOV: " << RAD2DEG(FOV) << " fovY: " << RAD2DEG(fovY) << " fovX: " << RAD2DEG(fovX) << " focal: " << focal << endl;
        }
    }
    else {
        sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

        if(m_framecount < 1) {
            cout << "u: " << camera_u << endl;
            cout << "v: " << camera_v << endl;
            cout << "w: " << camera_w << endl;
        }
        
        const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
                normalize( camera_u ),
                normalize( camera_v ),
                normalize( -camera_w ),
                camera_lookat);
        const optix::Matrix4x4 frame_inv = frame.inverse();
        // Apply camera rotation twice to match old SDK behavior
        const optix::Matrix4x4 trans   = frame*camera_rotate*camera_rotate*frame_inv;

        camera_eye    = optix::make_float3( trans*optix::make_float4( camera_eye,    1.0f ) );
        camera_lookat = optix::make_float3( trans*optix::make_float4( camera_lookat, 1.0f ) );
        camera_up     = optix::make_float3( trans*optix::make_float4( camera_up,     0.0f ) );

        sutil::calculateCameraVariables(
                camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
                camera_u, camera_v, camera_w, true );

        camera_rotate = optix::Matrix4x4::identity();
    }

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );

    // render
    context["frame"]->setUint( accumulation_frame++ );
    context->launch( 0, m_width, m_height );
    optix::Buffer buffer = context[ "output_buffer" ]->getBuffer();
    
    // Query buffer information
    RTsize buffer_width_rts, buffer_height_rts;
    buffer->getSize( buffer_width_rts, buffer_height_rts );
    uint32_t width  = static_cast<int>(buffer_width_rts);
    uint32_t height = static_cast<int>(buffer_height_rts);

    if(m_framecount < 1) {
        cout << width << " " << height << endl;
        cout << "eye: " << camera_eye << endl;
        cout << "u: " << camera_u << endl;
        cout << "v: " << camera_v << endl;
        cout << "w: " << camera_w << endl;
    }

    //RTformat buffer_format;
    //RT_CHECK_ERROR( rtBufferGetFormat( buffer->get(), &buffer_format ) );
    RTformat buffer_format = buffer->getFormat();
    const unsigned pboId = buffer->getGLBOId();

    if(pboId > 0) {
        if( !gl_screen_tex ) {
            glGenTextures( 1, &gl_screen_tex );
            glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

            // Change these to GL_LINEAR for super- or sub-sampling
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

            // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            cout << "gl_screen_tex: " << gl_screen_tex << endl;
        }

        glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

        // send PBO or host-mapped image data to texture
        
        GLvoid* imageData = 0;
        if( pboId )
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pboId );
        else
            imageData = buffer->map( 0, RT_BUFFER_MAP_READ );

        RTsize elmt_size = buffer->getElementSize();
        if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        GLenum pixel_format = glFormatFromBufferFormat(BUFFER_PIXEL_FORMAT_DEFAULT, buffer_format);

        if( buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, pixel_format, GL_UNSIGNED_BYTE, imageData);
        else if(buffer_format == RT_FORMAT_FLOAT4)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
        else if(buffer_format == RT_FORMAT_FLOAT3)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
        else if(buffer_format == RT_FORMAT_FLOAT)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
        else {
            cout << "Unkown buffer format" << endl;
            exit(1);
            // throw Exception( "Unknown buffer format" );
        }

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
        
        renderScreenQuadGL ( gl_screen_tex, 0, width, height );
    }

    else {
        GLenum gl_data_type;
        GLenum gl_format;
        switch (buffer_format) {
            case RT_FORMAT_UNSIGNED_BYTE4:
                gl_data_type = GL_UNSIGNED_BYTE;
                gl_format    = GL_BGRA;
                break;

            case RT_FORMAT_FLOAT:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_LUMINANCE;
                break;

            case RT_FORMAT_FLOAT3:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_RGB;
                break;

            case RT_FORMAT_FLOAT4:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_RGBA;
                break;

            default:
                fprintf(stderr, "Unrecognized buffer data type or format.\n");
                exit(2);
                break;
        }

        GLvoid* imageData = 0;
        RT_CHECK_ERROR( rtBufferMap( buffer->get(), &imageData ) );

        glDrawPixels(width, height, gl_format, gl_data_type, imageData);  // Using default glPixelStore unpack alignment of 4.

        // Now unmap the buffer
        RT_CHECK_ERROR( rtBufferUnmap( buffer->get() ) );
    }

	m_framecount++;
}

// for Omegalib CAVE
void OptixApp::display(const float cam_pos[3], const float cam_ori[4], const float head_off[3], 
                const float tl[3], const float bl[3], const float br[3]) {

    if(!m_initialized) return;

    optix::float3 camera_position = optix::make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
    optix::float4 camera_orientation = optix::make_float4(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3]);
    //optix::float3 head_offset = optix::make_float3(head_off[0], head_off[1], head_off[2]);
    optix::float3 head_offset = optix::make_float3(0, 2, 0);
    optix::float3 tile_tl = optix::make_float3(tl[0], tl[1], tl[2]);
    optix::float3 tile_bl = optix::make_float3(bl[0], bl[1], bl[2]);
    optix::float3 tile_br = optix::make_float3(br[0], br[1], br[2]);

    context["camera_position"]->setFloat( camera_position );
    context["camera_orientation"  ]->setFloat( camera_orientation );
    context["head_offset"  ]->setFloat( head_offset );
    context["tile_tl"  ]->setFloat( tile_tl );
    context["tile_bl"  ]->setFloat( tile_bl );
    context["tile_br"  ]->setFloat( tile_br );

    // render
    context["frame"]->setUint( accumulation_frame++ );
    context->launch( 0, m_width, m_height );
    optix::Buffer buffer = context[ "output_buffer" ]->getBuffer();
    
    // Query buffer information
    RTsize buffer_width_rts, buffer_height_rts;
    buffer->getSize( buffer_width_rts, buffer_height_rts );
    uint32_t width  = static_cast<int>(buffer_width_rts);
    uint32_t height = static_cast<int>(buffer_height_rts);

    // DEBUG
    if(m_framecount == 0) {
        cout << width << " " << height << endl;
        cout << "cam pos: " << camera_position << endl;
        cout << "cam ori: " << camera_orientation << endl;
        cout << "head offset: " << head_offset << endl;
        cout << "tile tl: " << tile_tl << " bl: " << tile_bl << " br: " << tile_br << endl;
    }
    
    RTformat buffer_format = buffer->getFormat();
    const unsigned pboId = buffer->getGLBOId();

    if(pboId > 0) {
        if( !gl_screen_tex ) {
            glGenTextures( 1, &gl_screen_tex );
            glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

            // Change these to GL_LINEAR for super- or sub-sampling
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

            // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            cout << "gl_screen_tex: " << gl_screen_tex << endl;
        }

        glBindTexture( GL_TEXTURE_2D, gl_screen_tex );

        // send PBO or host-mapped image data to texture
        
        GLvoid* imageData = 0;
        if( pboId )
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pboId );
        else
            imageData = buffer->map( 0, RT_BUFFER_MAP_READ );

        RTsize elmt_size = buffer->getElementSize();
        if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        GLenum pixel_format = glFormatFromBufferFormat(BUFFER_PIXEL_FORMAT_DEFAULT, buffer_format);

        if( buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, pixel_format, GL_UNSIGNED_BYTE, imageData);
        else if(buffer_format == RT_FORMAT_FLOAT4)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
        else if(buffer_format == RT_FORMAT_FLOAT3)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
        else if(buffer_format == RT_FORMAT_FLOAT)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
        else {
            cout << "Unkown buffer format" << endl;
            exit(1);
            // throw Exception( "Unknown buffer format" );
        }

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
        
        renderScreenQuadGL ( gl_screen_tex, 0, width, height );
    }

    else {
        GLenum gl_data_type;
        GLenum gl_format;
        switch (buffer_format) {
            case RT_FORMAT_UNSIGNED_BYTE4:
                gl_data_type = GL_UNSIGNED_BYTE;
                gl_format    = GL_BGRA;
                break;

            case RT_FORMAT_FLOAT:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_LUMINANCE;
                break;

            case RT_FORMAT_FLOAT3:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_RGB;
                break;

            case RT_FORMAT_FLOAT4:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_RGBA;
                break;

            default:
                fprintf(stderr, "Unrecognized buffer data type or format.\n");
                exit(2);
                break;
        }

        GLvoid* imageData = 0;
        RT_CHECK_ERROR( rtBufferMap( buffer->get(), &imageData ) );

        glDrawPixels(width, height, gl_format, gl_data_type, imageData);  // Using default glPixelStore unpack alignment of 4.

        // Now unmap the buffer
        RT_CHECK_ERROR( rtBufferUnmap( buffer->get() ) );
    }

    m_framecount++;
}

// ===== OPTIX ===========
std::string OptixApp::ptxPath( const std::string& cuda_file ) {
    return
        std::string(sutil::samplesPTXDir()) +
        "/" + std::string(SAMPLE_NAME) + "_generated_" +
        cuda_file +
        ".ptx";
}

optix::GeometryInstance OptixApp::createSphere(optix::Context context, optix::Material material,
                                    float3 center, float radius) {
    optix::Geometry sphere = context->createGeometry();
	sphere->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("sphere_intersect.cu");
	sphere->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	sphere->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "sphere_intersect_robust"));

	sphere["center"]->setFloat(center);
	sphere["radius"]->setFloat(radius);

	optix::GeometryInstance instance = context->createGeometryInstance(sphere, &material, &material + 1);
	return instance;
}

optix::GeometryInstance OptixApp::createQuad(optix::Context context,
                                    optix::Material material,
                                    float3 v1, float3 v2, float3 anchor, float3 n) {
    optix::Geometry quad = context->createGeometry();
	quad->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("quad_intersect.cu");
	quad->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	quad->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "intersect"));

	float3 normal = optix::normalize(optix::cross(v1, v2));
	float4 plane = optix::make_float4(normal, optix::dot(normal, anchor));
	v1 *= 1.0f / optix::dot(v1, v1);
	v2 *= 1.0f / optix::dot(v2, v2);
	quad["v1"]->setFloat(v1);
	quad["v2"]->setFloat(v2);
	quad["anchor"]->setFloat(anchor);
	quad["plane"]->setFloat(plane);

	optix::GeometryInstance instance = context->createGeometryInstance(quad, &material, &material + 1);
	return instance;
}

Buffer OptixApp::getOutputBuffer() {
    return context[ "output_buffer" ]->getBuffer();
}

void OptixApp::destroyContext() {
    if( context ) {
        context->destroy();
        context = 0;
    }
}

void OptixApp::createContext() {
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );

    // Note: this sample does not need a big stack size even with high ray depths, 
    // because rays are not shot recursively.
    context->setStackSize( 800 );

    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 3 );
    context["cutoff_color"]->setFloat( 0.0f, 0.0f, 0.0f );
    context["frame"]->setUint( 0u );
    context["scene_epsilon"]->setFloat( 1.e-3f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Accumulation buffer
    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT4, m_width, m_height);
    context["accum_buffer"]->set( accum_buffer );

    // Ray generation program
    std::string ptx_path( ptxPath( "path_trace_camera.cu" ) );
    Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXFile( ptx_path, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    ptx_path = ptxPath( "background.cu" );
    context->setMissProgram( 0, context->createProgramFromPTXFile( ptx_path, "miss" ) );
	const std::string texture_filename = std::string(sutil::samplesDir()) + "/data/CedarCity.hdr";
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texture_filename, optix::make_float3(1.0f)));

	Program prg;
	// BRDF sampling functions.
	m_bufferBRDFSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfSample = (int*) m_bufferBRDFSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Sample");
	brdfSample[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Sample");
	brdfSample[1] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Sample");
	brdfSample[2] = prg->getId();
	m_bufferBRDFSample->unmap();
	context["sysBRDFSample"]->setBuffer(m_bufferBRDFSample);
	
	// BRDF Eval functions.
	m_bufferBRDFEval = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfEval = (int*) m_bufferBRDFEval->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Eval");
	brdfEval[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Eval");
	brdfEval[1] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Eval");
	brdfEval[2] = prg->getId();
	m_bufferBRDFEval->unmap();
	context["sysBRDFEval"]->setBuffer(m_bufferBRDFEval);
	
	// BRDF Pdf functions.
	m_bufferBRDFPdf = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfPdf = (int*) m_bufferBRDFPdf->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Pdf");
	brdfPdf[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Pdf");
	brdfPdf[1] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Pdf");
	brdfPdf[2] = prg->getId();
	m_bufferBRDFPdf->unmap();
	context["sysBRDFPdf"]->setBuffer(m_bufferBRDFPdf);

	// Light sampling functions.
	m_bufferLightSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_LIGHT_INDICES);
	int* lightsample = (int*)m_bufferLightSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "sphere_sample");
	lightsample[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "quad_sample");
	lightsample[1] = prg->getId();
	m_bufferLightSample->unmap();
	context["sysLightSample"]->setBuffer(m_bufferLightSample);
}

optix::Material OptixApp::createMaterial(const MaterialParameter &mat, int index) {
    const std::string ptx_path = ptxPath( "hit_program.cu" );
	Program ch_program = context->createProgramFromPTXFile( ptx_path, "closest_hit" );
	Program ah_program = context->createProgramFromPTXFile(ptx_path, "any_hit");
	
	Material material = context->createMaterial();
	material->setClosestHitProgram( 0, ch_program );
	material->setAnyHitProgram(1, ah_program);
	
	material["materialId"]->setInt(index);
	material["programId"]->setInt(mat.brdf);

	return material;
}

optix::Material OptixApp::createLightMaterial(const LightParameter &mat, int index) {
    const std::string ptx_path = ptxPath("light_hit_program.cu");
	Program ch_program = context->createProgramFromPTXFile(ptx_path, "closest_hit");

	Material material = context->createMaterial();
	material->setClosestHitProgram(0, ch_program);

	material["lightMaterialId"]->setInt(index);

	return material;
}

void OptixApp::updateMaterialParameters(const std::vector<MaterialParameter> &materials) {
    MaterialParameter* dst = static_cast<MaterialParameter*>(m_bufferMaterialParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < materials.size(); ++i, ++dst) {
		MaterialParameter mat = materials[i];
		
		dst->color = mat.color;
		dst->emission = mat.emission;
		dst->metallic = mat.metallic;
		dst->subsurface = mat.subsurface;
		dst->specular = mat.specular;
		dst->specularTint = mat.specularTint;
		dst->roughness = mat.roughness;
		dst->anisotropic = mat.anisotropic;
		dst->sheen = mat.sheen;
		dst->sheenTint = mat.sheenTint;
		dst->clearcoat = mat.clearcoat;
		dst->clearcoatGloss = mat.clearcoatGloss;
		dst->brdf = mat.brdf;
		dst->albedoID = mat.albedoID;
	}
	m_bufferMaterialParameters->unmap();
}

void OptixApp::updateLightParameters(const std::vector<LightParameter> &lightParameters) {
    LightParameter* dst = static_cast<LightParameter*>(m_bufferLightParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < lightParameters.size(); ++i, ++dst) {
		LightParameter mat = lightParameters[i];

		dst->position = mat.position;
		dst->emission = mat.emission;
		dst->radius = mat.radius;
		dst->area = mat.area;
		dst->u = mat.u;
		dst->v = mat.v;
		dst->normal = mat.normal;
		dst->lightType = mat.lightType;
	}
	m_bufferLightParameters->unmap();
}

optix::Aabb OptixApp::createGeometry(optix::Group& top_group) {
    const std::string ptx_path = ptxPath( "triangle_mesh.cu" );

    top_group = context->createGroup();
    top_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

    int num_triangles = 0;
	size_t i,j;
    optix::Aabb aabb;
    {
        GeometryGroup geometry_group = context->createGeometryGroup();
        geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
        top_group->addChild( geometry_group );
		
        for (i = 0,j=0; i < scene->mesh_names.size(); ++i,++j) {
            OptiXMesh mesh;
            mesh.context = context;
            
            // override defaults
            mesh.intersection = context->createProgramFromPTXFile( ptx_path, "mesh_intersect_refine" );
            mesh.bounds = context->createProgramFromPTXFile( ptx_path, "mesh_bounds" );
            mesh.material = createMaterial(scene->materials[i], i);

            loadMesh( scene->mesh_names[i], mesh, scene->transforms[i] ); 
            geometry_group->addChild( mesh.geom_instance );

            aabb.include( mesh.bbox_min, mesh.bbox_max );

            std::cerr << scene->mesh_names[i] << ": " << mesh.num_triangles << std::endl;
            num_triangles += mesh.num_triangles;
        }
        std::cerr << "Total triangle count: " << num_triangles << std::endl;
    }
	//Lights
	{
		GeometryGroup geometry_group = context->createGeometryGroup();
		geometry_group->setAcceleration(context->createAcceleration("NoAccel"));
		top_group->addChild(geometry_group);
		
		for (i = 0; i < scene->lights.size(); ++i)
		{
			GeometryInstance instance;
			if (scene->lights[i].lightType == QUAD)
				instance = createQuad(context, createLightMaterial(scene->lights[i], i), scene->lights[i].u, scene->lights[i].v, scene->lights[i].position, scene->lights[i].normal);
			else if (scene->lights[i].lightType == SPHERE)
				instance = createSphere(context, createLightMaterial(scene->lights[i], i), scene->lights[i].position, scene->lights[i].radius);
			geometry_group->addChild(instance);
		}
		//GeometryInstance instance = createSphere(context, createMaterial(materials[j], j), optix::make_float3(150, 80, 120), 80);
		//geometry_group->addChild(instance);
	}

    context[ "top_object" ]->set( top_group ); 

    return aabb;
}


// ===== OPENGL ==========

void OptixApp::checkGL( char* msg ) {
	GLenum errCode;
    //const GLubyte* errString;
    errCode = glGetError();
    if (errCode != GL_NO_ERROR) {
		const char * message = "";
		switch( errCode )
		{
		case GL_INVALID_ENUM:
			message = "Invalid enum";
			break;
		case GL_INVALID_VALUE:
			message = "Invalid value";
			break;
		case GL_INVALID_OPERATION:
			message = "Invalid operation";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			message = "Invalid framebuffer operation";
			break;
		case GL_OUT_OF_MEMORY:
			message = "Out of memory";
			break;
		default:
			message = "Unknown error";
		}

        //printf ( "%s, ERROR: %s\n", msg, gluErrorString(errCode) );
		printf ( "%s %s\n", msg, message );
    }
}


void OptixApp::initGL() {
	initScreenQuadGL();
	glFinish();
}

void OptixApp::initScreenQuadGL() {
	int status;
	int maxLog = 65536, lenLog;
	char log[65536];

	// Create a screen-space shader
	m_screenquad_prog = (int)glCreateProgram();
	GLuint vShader = (int)glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vShader, 1, (const GLchar**)&g_screenquad_vert, NULL);
	glCompileShader(vShader);
	glGetShaderiv(vShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		glGetShaderInfoLog(vShader, maxLog, &lenLog, log);
		printf("*** Compile Error in init_screenquad vShader\n");
		printf("  %s\n", log);
	}

	GLuint fShader = (int)glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fShader, 1, (const GLchar**)&g_screenquad_frag, NULL);
	glCompileShader(fShader);
	glGetShaderiv(fShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		glGetShaderInfoLog(fShader, maxLog, &lenLog, log);
		printf("*** Compile Error in init_screenquad fShader\n");
		printf("  %s\n", log);
	}
	glAttachShader(m_screenquad_prog, vShader);
	glAttachShader(m_screenquad_prog, fShader);
	glLinkProgram(m_screenquad_prog);
	glGetProgramiv(m_screenquad_prog, GL_LINK_STATUS, &status);
	if (!status) {
		printf("*** Error! Failed to link in init_screenquad\n");
	}
	checkGL ( "glLinkProgram (init_screenquad)" );
	
	// Get texture parameter
	m_screenquad_utex1 = glGetUniformLocation (m_screenquad_prog, "uTex1" );
	m_screenquad_utex2 = glGetUniformLocation (m_screenquad_prog, "uTex2");
	m_screenquad_utexflags = glGetUniformLocation(m_screenquad_prog, "uTexFlags");
	m_screenquad_ucoords = glGetUniformLocation ( m_screenquad_prog, "uCoords" );
	m_screenquad_uscreen = glGetUniformLocation ( m_screenquad_prog, "uScreen" );


	// Create a screen-space quad VBO
	std::vector<nvVertex> verts;
	std::vector<nvFace> faces;
	verts.push_back(nvVertex(-1, -1, 0, -1, 1, 0));
	verts.push_back(nvVertex(1, -1, 0, 1, 1, 0));
	verts.push_back(nvVertex(1, 1, 0, 1, -1, 0));
	verts.push_back(nvVertex(-1, 1, 0, -1, -1, 0));
	faces.push_back(nvFace(0, 1, 2));
	faces.push_back(nvFace(2, 3, 0));

	glGenBuffers(1, (GLuint*)&m_screenquad_vbo[0]);
	glGenBuffers(1, (GLuint*)&m_screenquad_vbo[1]);
	checkGL("glGenBuffers (init_screenquad)");
	glGenVertexArrays(1, (GLuint*)&m_screenquad_vbo[2]);
	glBindVertexArray(m_screenquad_vbo[2]);
	checkGL("glGenVertexArrays (init_screenquad)");
	glBindBuffer(GL_ARRAY_BUFFER, m_screenquad_vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(nvVertex), &verts[0].x, GL_STATIC_DRAW_ARB);
	checkGL("glBufferData[V] (init_screenquad)");
	glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);				// pos
	glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);	// norm
	glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);	// texcoord
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenquad_vbo[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * 3 * sizeof(int), &faces[0].a, GL_STATIC_DRAW_ARB);
	checkGL("glBufferData[F] (init_screenquad)");
	glBindVertexArray(0);
}

void OptixApp::createScreenQuadGL ( int* glid, int w, int h ) {
	if ( *glid == -1 ) glDeleteTextures ( 1, (GLuint*) glid );
	glGenTextures ( 1, (GLuint*) glid );
	glBindTexture ( GL_TEXTURE_2D, *glid );
	checkGL ( "glBindTexture (createScreenQuadGL)" );
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 );	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);	
	glTexImage2D  ( GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	checkGL ( "glTexImage2D (createScreenQuadGL)" );
	glBindTexture ( GL_TEXTURE_2D, 0 );
}

void OptixApp::renderScreenQuadGL ( int glid1, int glid2, float x1, float y1, 
									float x2, float y2, char inv1, char inv2 ) {
	// Prepare pipeline
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(GL_FALSE);
	// Select shader	
	glBindVertexArray(m_screenquad_vbo[2]);
	glUseProgram(m_screenquad_prog);
	checkGL("glUseProgram");
	// Select VBO	
	glBindBuffer(GL_ARRAY_BUFFER, m_screenquad_vbo[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenquad_vbo[1]);
	checkGL("glBindBuffer");
	// Select texture
	
	//glEnable ( GL_TEXTURE_2D );
	
	glProgramUniform4f ( m_screenquad_prog, m_screenquad_ucoords, x1, y1, x2, y2 );
	glProgramUniform2f ( m_screenquad_prog, m_screenquad_uscreen, x2, y2 );
	glActiveTexture ( GL_TEXTURE0 );
	glBindTexture ( GL_TEXTURE_2D, glid1 );
    checkGL("glBindTexture");
	
	glProgramUniform1i(m_screenquad_prog, m_screenquad_utex1, 0);
	int flags = 0;
	if (inv1 > 0) flags |= 1;												// y-invert tex1

	if (glid2 >= 0) {
		flags |= 2;															// enable tex2 compositing
		if (inv2 > 0) flags |= 4;											// y-invert tex2
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, glid2);
		glProgramUniform1i(m_screenquad_prog, m_screenquad_utex2, 1);
	}

	glProgramUniform1i(m_screenquad_prog, m_screenquad_utexflags, flags );	

	// Draw
	glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1);
	
	checkGL("glDraw");
	glUseProgram(0);

	glDepthMask(GL_TRUE);
}

void OptixApp::renderScreenQuadGL( int glid, char inv1, int w, int h ) {
	renderScreenQuadGL ( glid, -1, (float)0, (float)0, (float)w, (float)h, inv1, 0); 
}