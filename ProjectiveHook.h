#ifndef MPMHOOK_H
#define MPMHOOK_H
#include "PhysicsHook.h"
#include <iostream>
#include <Eigen/Core>
#include <string>
#include "igl/triangulated_grid.h"
#include <Eigen/StdVector>

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);

typedef std::tuple<int, int> Edge;

struct MouseEvent
{
    enum METype {
        ME_CLICKED,
        ME_RELEASED,
        ME_DRAGGED
    };

    METype type;
    int vertex;
    int button;
	int fid;
    Eigen::Vector3d pos;
};


class ProjectiveHook : public PhysicsHook
{
public:
	ProjectiveHook();

    virtual void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu);
    virtual void tick();
    virtual void initSimulation();

    virtual void updateGeometry() {
		renderQ = Q;
		renderF = F;
    }

    virtual bool simulationStep();
    virtual void renderGeometry(igl::opengl::glfw::Viewer &viewer) {
		viewer.data().set_mesh(renderQ, renderF);
    }

    virtual bool mouseClicked(igl::opengl::glfw::Viewer &viewer, int button);
    virtual bool mouseReleased(igl::opengl::glfw::Viewer &viewer,  int button);

    private:

    Eigen::MatrixXd renderQ;
    Eigen::MatrixXi renderF;
	Eigen::MatrixXd Q;					// current positions
	Eigen::MatrixXd Q0;					// initial positions
	Eigen::MatrixXd Qdot;				// current velocities
	Eigen::MatrixXi F;					// triangle faces
	Eigen::SparseMatrix<double> M;		// mass matrix
	Eigen::SparseMatrix<double> Minv;	// mass matrix inverse
	Eigen::MatrixXd fExt;               // external forces

	std::vector<Eigen::SparseMatrix<double>, Eigen::aligned_allocator<Eigen::SparseMatrix<double>>> SList;
	std::vector<Eigen::SparseMatrix<double>, Eigen::aligned_allocator<Eigen::SparseMatrix<double>>> SPosList;
	std::vector<Eigen::Matrix<double, 6, 9>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 9>>> AList;
	std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> DmInvList;


    double dt;
    double gravity;
	double wtri;
	double min_sval;
	double max_sval;
    std::string meshFile;
    std::mutex mouseMutex;
    std::vector<MouseEvent> mouseEvents;
    Eigen::Vector3d curPos;		// the current position of the mouse cursor in 3D
    int clickedVertex;			// the currently selected vertex (-1 if no vertex)
	double clickedz;
	int clickedFace;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

	std::vector<int> positionIndices;

};

#endif // MPMHOOK_H
