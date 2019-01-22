/*Copyright (c) 2016 Miles Macklin

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.*/

#include"sceneLoader.h"

static const int kMaxLineLength = 2048;

Scene* LoadScene(const char* filename, int w, int h)
{
	Scene *scene = new Scene;
	int tex_id = 0;
	FILE* file = fopen(filename, "r");

	if (!file)
	{
		printf("Couldn't open %s for reading.", filename);
		return NULL;
	}

	std::map<std::string, MaterialParameter> materials_map;
	std::map<std::string, int> texture_ids;

	char line[kMaxLineLength];

	while (fgets(line, kMaxLineLength, file))
	{
		// skip comments
		if (line[0] == '#')
			continue;

		// name used for materials and meshes
		char name[kMaxLineLength] = { 0 };


		//--------------------------------------------
		// Material

		if (sscanf(line, " material %s", name) == 1)
		{
			printf("%s", line);

			MaterialParameter material;
			char tex_name[kMaxLineLength] = "None";

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " name %s", name);
				sscanf(line, " color %f %f %f", &material.color.x, &material.color.y, &material.color.z);
				sscanf(line, " albedoTex %s", &tex_name);
				sscanf(line, " emission %f %f %f", &material.emission.x, &material.emission.y, &material.emission.z);

				sscanf(line, " metallic %f", &material.metallic);
				sscanf(line, " subsurface %f", &material.subsurface);
				sscanf(line, " specular %f", &material.specular);
				sscanf(line, " specularTint %f", &material.specularTint);
				sscanf(line, " roughness %f", &material.roughness);
				sscanf(line, " anisotropic %f", &material.anisotropic);
				sscanf(line, " sheen %f", &material.sheen);
				sscanf(line, " sheenTint %f", &material.sheenTint);
				sscanf(line, " clearcoat %f", &material.clearcoat);
				sscanf(line, " clearcoatGloss %f", &material.clearcoatGloss);
				sscanf(line, " brdf %i", &material.brdf);
			}

			// Check if texture is already loaded
			if (texture_ids.find(tex_name) != texture_ids.end()) // Found Texture
			{
				material.albedoID = texture_ids[tex_name];
			}
			else if(strcmp(tex_name, "None") != 0)
			{
				tex_id++;
				texture_ids[tex_name] = tex_id;
				scene->texture_map[tex_id - 1] = tex_name;
				material.albedoID = tex_id;
			}

			// add material to map
			materials_map[name] = material;
		}

		//--------------------------------------------
		// Light

		if (strstr(line, "light"))
		{
			LightParameter light;
			optix::float3 v1, v2;
			char light_type[20] = "None";

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " position %f %f %f", &light.position.x, &light.position.y, &light.position.z);
				sscanf(line, " emission %f %f %f", &light.emission.x, &light.emission.y, &light.emission.z);
				sscanf(line, " normal %f %f %f", &light.normal.x, &light.normal.y, &light.normal.z);

				sscanf(line, " radius %f", &light.radius);
				//sscanf(line, " u %f %f %f", &light.v1.x, &light.v1.y, &light.v1.z);
				//sscanf(line, " v %f %f %f", &light.v2.x, &light.v2.y, &light.v2.z);
				sscanf(line, " v1 %f %f %f", &v1.x, &v1.y, &v1.z);
				sscanf(line, " v2 %f %f %f", &v2.x, &v2.y, &v2.z);
				sscanf(line, " type %s", light_type);
			}

			if (strcmp(light_type, "Quad") == 0)
			{
				light.lightType = QUAD;
				light.u = v1 - light.position;
				light.v = v2 - light.position;
				light.area = optix::length(optix::cross(light.u, light.v));
				light.normal = optix::normalize(optix::cross(light.u, light.v));
			}
			else if (strcmp(light_type, "Sphere") == 0)
			{
				light.lightType = SPHERE;
				light.normal = optix::normalize(light.normal);
				light.area = 4.0f * M_PIf * light.radius * light.radius;
			}

			scene->lights.push_back(light);
		}

		//--------------------------------------------
		// Properties

		//Defaults
		Properties prop;
		prop.width = 1280;
		prop.height = 720;
		scene->properties = prop;

		if (strstr(line, "properties"))
		{

			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				sscanf(line, " width %i", &prop.width);
				sscanf(line, " height %i", &prop.height);
			}
			prop.width = w;
			prop.height = h;
			scene->properties = prop;
		}

		//--------------------------------------------
		// Mesh

		if (strstr(line, "mesh"))
		{
			while (fgets(line, kMaxLineLength, file))
			{
				// end group
				if (strchr(line, '}'))
					break;

				int count = 0;

				char path[2048];

				if (sscanf(line, " file %s", path) == 1)
				{
					const optix::Matrix4x4 xform = optix::Matrix4x4::identity();// optix::Matrix4x4::rotate(-M_PIf / 2.0f, optix::make_float3(0.0f, 1.0f, 0.0f)
					scene->mesh_names.push_back(std::string(sutil::samplesDir()) + "/data/" + path);
					scene->transforms.push_back(xform);
				}

				if (sscanf(line, " material %s", path) == 1)
				{
					// look up material in dictionary
					if (materials_map.find(path) != materials_map.end())
					{
						scene->materials.push_back(materials_map[path]);
					}
					else
					{
						printf("Could not find material %s\n", path);
					}
				}
			}
		}
	}
	return scene;
}