#include "CLI11.hpp"

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gldraw.h>

#include <pangolin/pangolin.h>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <filesystem>

namespace fs = std::filesystem;
namespace AA = Alembic::Abc;
namespace AAG = Alembic::AbcGeom;

const std::string depth_shader = R"Shader(
@start vertex
#version 120
uniform mat4 MV;
uniform mat4 P;
attribute vec3 position;
varying float depth;

void main() {
    vec4 position_cam = MV * vec4(position, 1);
    gl_Position = P * position_cam;
    // depth = length(position_cam.xyz); // distance
    depth = position_cam.z; // z
}

@start fragment
#version 120
varying float depth;
void main() {
    gl_FragColor = vec4(depth, depth, depth, 1.0);
}
)Shader";


struct Mesh {
    std::vector<float> vertices;
    std::vector<uint32_t> indexes;
};

struct MeshOpenGL {

    MeshOpenGL(const MeshOpenGL&) = delete;
    MeshOpenGL& operator=(const MeshOpenGL&) = delete;

    MeshOpenGL(std::shared_ptr<Mesh> mesh){
        data_cpu = mesh;
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        bind();
        glBufferData(GL_ARRAY_BUFFER, 4 * mesh->vertices.size(), &(mesh->vertices[0]), GL_STATIC_DRAW);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * mesh->indexes.size(), &(mesh->indexes[0]), GL_STATIC_DRAW);
        unbind();
    }

    virtual ~MeshOpenGL(){
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ibo);
    }

    void bind(){
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    }

    void unbind(){
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    GLuint vbo = 0;
    GLuint ibo = 0;
    std::shared_ptr<Mesh> data_cpu;
};

std::shared_ptr<Mesh> load_obj(const fs::path& file_path) {
    if(!fs::exists(file_path)){
        std::cerr << "File does not exist: " << file_path << std::endl;
        exit(1);
    }
    std::ifstream indata;
    indata.open(file_path);
    std::string line;
    std::string property;
    uint32_t i0, i1, i2;
    float vx, vy, vz;
    Mesh mesh;

    while (std::getline(indata, line)) {
        std::stringstream line_stream(line);
        std::getline(line_stream, property, ' ');
        if (property == "f") {
            line_stream >> i0 >> i1 >> i2;
            mesh.indexes.push_back(i0-1);
            mesh.indexes.push_back(i1-1);
            mesh.indexes.push_back(i2-1);
        } else if (property == "v") {
            line_stream >> vx >> vy >> vz;
            mesh.vertices.push_back(vx);
            mesh.vertices.push_back(vy);
            mesh.vertices.push_back(vz);
        } else if (property == "#") {
            // ignore comments
        } else if (property == "\r") {
            // ignore windows line ending
        } else {
            std::cerr << "Unsupported obj format (property '" << property << "'), expect issues." << std::endl;
        }
    }
    return std::make_shared<Mesh>(std::move(mesh));
}

std::shared_ptr<Mesh> load_mesh_sample(const AAG::IPolyMeshSchema::Sample& mesh_sample){
    Mesh mesh;
    for(size_t i=0; i < mesh_sample.getPositions()->size(); i++){
        auto& position = (*(mesh_sample.getPositions()))[i];
        mesh.vertices.push_back(position[0]);
        mesh.vertices.push_back(position[1]);
        mesh.vertices.push_back(position[2]);
    }

    for(size_t i=0; i < mesh_sample.getFaceIndices()->size(); i+=3){
        mesh.indexes.push_back((*(mesh_sample.getFaceIndices()))[i]);
        mesh.indexes.push_back((*(mesh_sample.getFaceIndices()))[i+1]);
        mesh.indexes.push_back((*(mesh_sample.getFaceIndices()))[i+2]);
    }
    return std::make_shared<Mesh>(std::move(mesh));
}


struct CameraData {

    // camera name
    std::string name;

    // resolution
    size_t width;
    size_t height;

    // extrinsics
    Eigen::Matrix4f transform_cam2world;

    // intrinsics
    float fx;
    float fy;
    float cx;
    float cy;
};


std::vector<CameraData> read_camera_data(const std::string& file_path){
    if(!fs::exists(file_path)){
        std::cerr << "File does not exist: " << file_path << std::endl;
        exit(2);
    }
    std::vector<CameraData> camera_data;

    std::ifstream indata;
    indata.open(file_path);
    std::string line;

    // skip header
    std::getline(indata, line);

    // read lines / cameras
    while (std::getline(indata, line)) {

        std::stringstream line_stream(line);
        auto read_next = [&line_stream](){
            std::string property;
            std::getline(line_stream, property, ',');
            return property;
        };

        std::string name;
        std::getline(line_stream, name, ',');
        size_t w = std::stoul(read_next());
        size_t h = std::stoul(read_next());
        float rx = std::stof(read_next());
        float ry = std::stof(read_next());
        float rz = std::stof(read_next());
        float tx = std::stof(read_next());
        float ty = std::stof(read_next());
        float tz = std::stof(read_next());
        float fx = std::stof(read_next());
        float fy = std::stof(read_next());
        float cx = std::stof(read_next());
        float cy = std::stof(read_next());

        Eigen::Matrix4f transform_cam2world = Eigen::Matrix4f::Identity();
        Eigen::Vector3f axis_angle(rx, ry, rz);
        transform_cam2world.topLeftCorner<3, 3>() = Eigen::AngleAxisf(axis_angle.norm(), axis_angle.normalized()).matrix();
        transform_cam2world.topRightCorner<3, 1>() << tx, ty, tz;
        camera_data.emplace_back(CameraData{name, w, h, transform_cam2world, fx, fy, cx, cy});
    }
    return camera_data;
}


struct Renderbuffer {
    Renderbuffer(GLint width, GLint height) :
        color_buffer(width, height, GL_RGBA32F, true, 0, GL_RGBA, GL_FLOAT),
        depth_buffer(),
        fbo_buffer(color_buffer, depth_buffer){
    }

    pangolin::GlTexture color_buffer;
    pangolin::GlRenderBuffer depth_buffer;
    pangolin::GlFramebuffer fbo_buffer;
};


int main(int argc, const char** argv)
{
    std::vector<fs::path> mesh_files;
    std::set<std::string> camera_names;
    fs::path path_alembic;
    std::string scene_description = "";
    fs::path path_calibration;
    fs::path output_folder;
    bool render_depth = false;
    bool render_mask = false;
    bool headless = false;
    const std::string application_name = "MultiViewRenderer";

    CLI::App app{application_name};
    // app.add_option("--mesh_files", mesh_files, "List of meshes, like 'frames/*.obj' or '/path/mesh001.obj /path/mesh002.obj ...'")->required();
    app.add_option("--alembic", path_alembic, "Input animation alembic (*.abc) file.")->required();
    // app.add_option("--obj", path_calibration, "Path to calibration csv file.");
    app.add_option("--csv", path_calibration, "Path to calibration csv file.")->required();
    app.add_option("--output", output_folder, "Output alembic file path, like '/path/out.abc'")->required();
    app.add_flag("--depth", render_depth, "If flag is set, render depth maps");
    app.add_flag("--mask", render_mask, "If flag is set, render masks");
    app.add_flag("--headless", headless, "If flag is set, use headless mode.");
    app.add_option("--cameras", camera_names, "Specify cameras to render (alternatively all cameras will be rendered).");
    CLI11_PARSE(app, argc, argv);

    if (!render_depth && !render_mask){
        std::cout << "Neither --depth nor --mask was set, abort." << std::endl;
        exit(3);
    }

    if (headless){
        pangolin::CreateWindowAndBind("render_window", 1, 1, pangolin::Params({{"scheme", "headless"}}));
    } else {
        pangolin::CreateWindowAndBind("render_window", 1, 1);
    }

    pangolin::GlSlProgram prog;
    prog.AddShader( pangolin::GlSlAnnotatedShader, depth_shader );
    prog.Link();
    prog.Bind();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Read and filter cameras, create output directories
    std::vector<CameraData> cameras = read_camera_data(path_calibration);
    if (camera_names.size() > 0){
        cameras.erase(std::remove_if(
            cameras.begin(),
            cameras.end(),
            [&](const CameraData& cam){ return camera_names.find(cam.name) == camera_names.end(); }
        ), cameras.end());
    }
    fs::create_directories(output_folder);
    for(auto& camera : cameras){
        fs::create_directories(output_folder / camera.name);
    }

    // TODO add support for rendering sequences of meshes, also support ABC
    // if(mesh_files.size() != 1){
    //     std::cerr << "currently only single obj files are supported." << std::endl;
    //     exit(1);
    // }


    std::map<std::pair<GLint, GLint>, std::shared_ptr<Renderbuffer>> renderbuffers;
    auto get_renderbuffer = [&renderbuffers](GLint width, GLint height) {
        std::pair<GLint, GLint> key = std::make_pair(width, height);
        auto rb = renderbuffers.find(key);
        if (rb != renderbuffers.end()){
            return rb->second;
        }
        auto new_renderbuffer = std::make_shared<Renderbuffer>(width, height);
        renderbuffers.emplace(key, new_renderbuffer);
        return new_renderbuffer;
    };


    AA::IArchive archive( Alembic::AbcCoreOgawa::ReadArchive(), path_alembic );
    AAG::IPolyMesh poly_mesh(AA::IObject( archive, AA::kTop ), "object" );
    AAG::IPolyMeshSchema& mesh_alembic = poly_mesh.getSchema();
    AAG::IPolyMeshSchema::Sample mesh_samp;

    size_t animation_legnth = mesh_alembic.getNumSamples();

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    for(size_t animation_idx = 0; animation_idx < animation_legnth; animation_idx++){

        mesh_alembic.get( mesh_samp, animation_idx );
        MeshOpenGL mesh(load_mesh_sample(mesh_samp));

        std::cout << "Rendering animation at frame: " << animation_idx << std::endl;

        // MeshOpenGL mesh(load_obj(mesh_files[0]));

        mesh.bind();
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        for(CameraData& camera : cameras){
            pangolin::OpenGlMatrix projection_matrix;
            projection_matrix = pangolin::ProjectionMatrixRDF_BottomLeft(
                        GLint(camera.width),
                        GLint(camera.height),
                        pangolin::GLprecision(camera.fx * camera.width),
                        pangolin::GLprecision(camera.fy * camera.height),
                        pangolin::GLprecision(camera.cx * camera.width),
                        pangolin::GLprecision(camera.cy * camera.height),
                        0.01f,
                        100.0f
                        );
            Eigen::Matrix4f mv = camera.transform_cam2world.inverse();

            prog.SetUniform("MV", mv);
            prog.SetUniform("P", projection_matrix);

            std::shared_ptr<Renderbuffer> renderbuffer = get_renderbuffer(camera.width, camera.height);
            renderbuffer->fbo_buffer.Bind();

            // Clear screen and activate view to render into
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glViewport(0, 0, camera.width, camera.height);

            glDrawElements(GL_TRIANGLES, mesh.data_cpu->indexes.size(), GL_UNSIGNED_INT, nullptr);
            pangolin::FinishFrame();

            renderbuffer->fbo_buffer.Unbind();

            // Export frames as exr / png
            pangolin::TypedImage image;
            renderbuffer->color_buffer.Download(image);
            cv::Mat img(image.h, image.w, CV_32FC4, static_cast<void*>(image.ptr));
            cv::Mat channel(image.h, image.w, CV_32FC1);
            std::stringstream index_stream;
            index_stream << std::setfill('0') << std::setw(6) << animation_idx;
            if (render_depth){
                cv::extractChannel(img, channel, 0);
                cv::imwrite(output_folder / camera.name / (camera.name + "_" + index_stream.str() + "_depth.exr"), channel);
            }
            if (render_mask){
                cv::extractChannel(img, channel, 3);
                cv::imwrite(output_folder / camera.name / (camera.name + "_" + index_stream.str() + "_mask.png"), channel == 1);
            }
        }
        mesh.unbind();
    }

    return 0;
}
