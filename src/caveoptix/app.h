#ifndef __OPTIX_APP_H
#define __OPTIX_APP_H

#ifdef OMEGALIB_MODULE
#include <omegaGl.h>
#else
#include "stdapp/GLInclude.h"
#endif

#include "sutil.h"
#include "commonStructs.h"
#include "random.h"
#include "properties.h"
#include "light_parameters.h"
#include "material_parameters.h"
#include "sceneLoader.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <string>

struct nvVertex {
	nvVertex(float x1, float y1, float z1, float tx1, float ty1, float tz1) { x=x1; y=y1; z=z1; tx=tx1; ty=ty1; tz=tz1; }
	float	x, y, z;
	float	nx, ny, nz;
	float	tx, ty, tz;
};
struct nvFace {
	nvFace(unsigned int x1, unsigned int y1, unsigned int z1) { a=x1; b=y1; c=z1; }
	unsigned int  a, b, c;
};

typedef optix::float3 float3;
typedef optix::float4 float4;

enum bufferPixelFormat
{
    BUFFER_PIXEL_FORMAT_DEFAULT, // The default depending on the buffer type
    BUFFER_PIXEL_FORMAT_RGB,     // The buffer is RGB or RGBA
    BUFFER_PIXEL_FORMAT_BGR,     // The buffer is BGR or BGRA
};

class OptixApp {

protected:
    // OpenGL
    void checkGL( char* msg );
    void initGL();
    void initScreenQuadGL();
    void createScreenQuadGL ( int* glid, int w, int h );
    void renderScreenQuadGL ( int glid1, int glid2, float x1, float y1, float x2, float y2, char inv1, char inv2 );
    void renderScreenQuadGL( int glid, char inv1, int w, int h );

    // optix
    std::string ptxPath( const std::string& cuda_file );
    optix::GeometryInstance createSphere(optix::Context context, optix::Material material,
	                                    float3 center, float radius);
    optix::GeometryInstance createQuad(optix::Context context,
	                                    optix::Material material,
	                                    float3 v1, float3 v2, float3 anchor, float3 n);
    optix::Buffer getOutputBuffer();
    void destroyContext();
    void createContext();
    optix::Material createMaterial(const MaterialParameter &mat, int index);
    optix::Material createLightMaterial(const LightParameter &mat, int index);
    void updateMaterialParameters(const std::vector<MaterialParameter> &materials);
    void updateLightParameters(const std::vector<LightParameter> &lightParameters);
    optix::Aabb createGeometry(optix::Group& top_group);
    optix::TextureSampler createSamplerFromFile(const std::string& filename, const optix::float3& default_color);

public:
    OptixApp(std::string scene_file);
    ~OptixApp();

    void init(int w, int h);
    void display(const float V[16], const float P[16], const float pos[3]);
    void display(const float cam_pos[3], const float cam_ori[4], const float head_offset[3], 
                const float tile_tl[3], const float tile_bl[3], const float tile_br[3]);
    void resetDraw() { accumulation_frame = 0; }

    
private:
    bool    m_initialized;
    int     m_framecount;

    int     m_width, m_height;
    int		m_screenquad_prog;
    int		m_screenquad_vshader;
    int		m_screenquad_fshader;
    int		m_screenquad_vbo[3];
    int		m_screenquad_utex1;
    int		m_screenquad_utex2;
    int		m_screenquad_utexflags;
    int		m_screenquad_ucoords;
    int		m_screenquad_uscreen;

    unsigned int	gl_screen_tex; 

    //Optix
    std::string     scene_file;
    optix::Context  context;
    bool            use_pbo;
    std::string     texture_path;
    
    // Camera state (not used with Omegalib)
    optix::float3       camera_up;
    optix::float3       camera_lookat;
    optix::float3       camera_eye;
    optix::Matrix4x4    camera_rotate;
    
    // from optixPathTracer
    optix::Buffer       m_bufferBRDFSample;
    optix::Buffer       m_bufferBRDFEval;
    optix::Buffer       m_bufferBRDFPdf;

    optix::Buffer       m_bufferLightSample;
    optix::Buffer       m_bufferMaterialParameters;
    optix::Buffer       m_bufferLightParameters;

    unsigned int        accumulation_frame;
    float               transmittance_log_scale;
    int                 max_depth;
    double              elapsedTime;
    double              lastTime;

    Properties          properties;
    Scene*              scene;
};


#endif