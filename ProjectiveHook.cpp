
#include "ProjectiveHook.h"
#include "Util.h"
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/polar_dec.h>
#include <igl/readPLY.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <string>
#include <omp.h>
#include <cmath>

using namespace Eigen;

ProjectiveHook::ProjectiveHook() : PhysicsHook()
{
    meshFile = "../square_mesh_med.obj";
    dt = 0.05;
    gravity = -0.25;
	wtri = 1;
	min_sval = 0.85;
	max_sval = 1.1;
	wpos = 50.0;
	punch = -5.0;
}


void ProjectiveHook::drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu)
{
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("Timestep", &dt);
		ImGui::InputDouble("Gravity", &gravity);
		ImGui::InputDouble("min sval", &min_sval);
		ImGui::InputDouble("max sval", &max_sval);
		ImGui::InputDouble("w_tri", &wtri);
		ImGui::InputDouble("w_pos", &wpos);
		ImGui::InputDouble("mike tysons", &punch);
		ImGui::InputText("Mesh file", meshFile);

    }
    //const char* listbox_items[] = { "Inferno", "Jet", "Magma", "Parula", "Plasma", "Viridis"};
    //if (ImGui::CollapsingHeader("Render Options", ImGuiTreeNodeFlags_DefaultOpen))
    //{
    //    ImGui::ListBox("Render color", &render_color, listbox_items, IM_ARRAYSIZE(listbox_items), 4);
    //}
}

bool ProjectiveHook::mouseClicked(igl::opengl::glfw::Viewer &viewer, int button) {
    render_mutex.lock();

    MouseEvent me;
    me.button = button;
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
      viewer.core().proj, viewer.core().viewport, Q, F, fid, bc)) {
        me.type = MouseEvent::ME_CLICKED;
		me.fid = fid;
    } else {
        me.type = MouseEvent::ME_RELEASED;
    }
    render_mutex.unlock();
    mouseMutex.lock();
    mouseEvents.push_back(me);
    mouseMutex.unlock();
    return false;
}

bool ProjectiveHook::mouseReleased(igl::opengl::glfw::Viewer &viewer, int button) {
    MouseEvent me;
    me.type = MouseEvent::ME_RELEASED;
    mouseMutex.lock();
    mouseEvents.push_back(me);
    mouseMutex.unlock();
    return false;
}

void ProjectiveHook::tick()
{
    mouseMutex.lock();
    for (MouseEvent me : mouseEvents) {
        if (me.type == MouseEvent::ME_CLICKED) {
            curPos = me.pos;
            clickedVertex = me.vertex;
            //button_ = me.button;
			clickedFace = me.fid;
			std::cout << "clicked: " << clickedFace << std::endl;
        }
        if (me.type == MouseEvent::ME_RELEASED) {
            clickedVertex = -1;
			clickedFace = -1;
        }
    }
    mouseEvents.clear();
    mouseMutex.unlock();
}

void ProjectiveHook::initSimulation()
{
    Eigen::initParallel();
	Util::readMesh(meshFile, Q, F);
	//igl::triangulated_grid(16,16,Q,F);

	Q0 = Q; //TODO remove z column?
	
	SparseMatrix<double> Mtmp;
	igl::massmatrix(Q, F, igl::MASSMATRIX_TYPE_DEFAULT, Mtmp);
	std::vector<Triplet<double>> Mtriplets;

	for (int i = 0; i < Mtmp.rows(); ++i) {
		Mtriplets.emplace_back(Triplet<double>(Mtmp.rows() * 0 + i, Mtmp.rows() * 0 + i, 1));//  Mtmp.coeff(i, i)));
		Mtriplets.emplace_back(Triplet<double>(Mtmp.rows() * 1 + i, Mtmp.rows() * 1 + i, 1));// Mtmp.coeff(i, i)));
		Mtriplets.emplace_back(Triplet<double>(Mtmp.rows() * 2 + i, Mtmp.rows() * 2 + i, 1));// Mtmp.coeff(i, i)));
	}
	M.resize(Mtmp.rows() * 3, Mtmp.rows() * 3);
	M.setFromTriplets(Mtriplets.begin(), Mtriplets.end());

	igl::invert_diag(M, Minv);
	IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	//std::cout << M.toDense().format(CleanFmt) << std::endl;

	Qdot.resizeLike(Q);
	Qdot.setZero();

	fExt.resizeLike(Q);
	fExt.setZero();
	fExt.col(1).setConstant(gravity);

	std::cout << F.format(CleanFmt) << std::endl;
	std::cout << Q.format(CleanFmt) << std::endl;

	//std::cout << Q.format(CleanFmt) << std::endl;

	// Global solve LHS
	SparseMatrix<double> LHS = M / (dt*dt);

	// Generate selection matrix
	for (int i = 0; i < F.rows(); ++i) {
		SparseMatrix<double> S(Q.rows() * 3, 9);
		std::vector<Triplet<double>> coefficients;

		for (int j = 0; j < 3; ++j) {
			// note: Eigen is column-major by default
			coefficients.emplace_back(Triplet<double>(Q.rows() * 0 + F(i, j), 3 * j + 0, 1));
			coefficients.emplace_back(Triplet<double>(Q.rows() * 1 + F(i, j), 3 * j + 1, 1));
			coefficients.emplace_back(Triplet<double>(Q.rows() * 2 + F(i, j), 3 * j + 2, 1));
		}
		S.setFromTriplets(coefficients.begin(), coefficients.end());

		Vector2d p0 = Q.row(F(i, 0)).head(2);
		Vector2d p1 = Q.row(F(i, 1)).head(2);
		Vector2d p2 = Q.row(F(i, 2)).head(2);

		// [ x2 - x0 | x1 - x0 ]
		Matrix2d Dm;
		Dm << p1(0) - p0(0), p2(0) - p0(0),
			  p1(1) - p0(1), p2(1) - p0(1);
		Matrix2d Dm_inv = Dm.inverse();

		// A matrix
		Matrix<double, 6, 9> A;
		A.setZero();
		for (int j = 0; j < 3; ++j) {
			A(j*2,   0+j) = -Dm_inv(0, 0) - Dm_inv(1, 0);
			A(j*2,   3+j) = Dm_inv(0, 0);
			A(j*2,   6+j) = Dm_inv(1, 0);
			A(j*2+1, 0+j) = -Dm_inv(0, 1) - Dm_inv(1, 1);
			A(j*2+1, 3+j) = Dm_inv(0, 1);
			A(j*2+1, 6+j) = Dm_inv(1, 1);
		}
		
		Matrix<double, 9, 6> AT = A.transpose();
		SparseMatrix<double> ST = S.transpose();

		SList.emplace_back(S);
		AList.emplace_back(A);
		DmInvList.emplace_back(Dm_inv);
		LHS += wtri * (S.toDense() * AT * A * ST.toDense()).sparseView();
	}

	positionIndices = { 0 , 2 };
	// Position constraints baby
	for (int i = 0; i < positionIndices.size(); ++i) {
		SparseMatrix<double> S(Q.rows() * 3, 3);
		std::vector<Triplet<double>> coefficients;
		coefficients.emplace_back(Triplet<double>(Q.rows() * 0 + positionIndices[i], 0, 1));
		coefficients.emplace_back(Triplet<double>(Q.rows() * 1 + positionIndices[i], 1, 1));
		coefficients.emplace_back(Triplet<double>(Q.rows() * 2 + positionIndices[i], 2, 1));
		S.setFromTriplets(coefficients.begin(), coefficients.end());
		SPosList.emplace_back(S);
		LHS += wpos * (S * S.transpose());
	}

	solver.compute(LHS);
    //igl::triangulated_grid(resolution_,resolution_,grid_V,grid_F);
}
bool ProjectiveHook::simulationStep() {
    bool failure = false; // Used to exit simulation loop when we hit a NaN.
	
	if (clickedFace != -1) {
		fExt(F(clickedFace, 0), 2) = punch;
		fExt(F(clickedFace, 1), 2) = punch;
		fExt(F(clickedFace, 2), 2) = punch;
	}

	// compute s_n
	Map<RowVectorXd> q(Q.data(), Q.size());
	Map<RowVectorXd> qdot(Qdot.data(), Qdot.size());
	Map<RowVectorXd> ext(fExt.data(), fExt.size());
	MatrixXd sn = q.transpose() + dt * qdot.transpose() + (dt * dt) * Minv * ext.transpose();
	MatrixXd qn = sn;

	std::vector<Matrix<double, 6, 1>, aligned_allocator<Matrix<double, 6, 1>>> BpList(F.rows());

	for (int outer = 0; outer < 1; ++outer) {
		// Local solve
#pragma omp parallel for
		for (int i = 0; i < F.rows(); ++i) {
			// Compute deformation gradient
			Matrix<double, 3, 2> XfXg;

			// First compute Xf
			XfXg << qn(               F(1)) - qn(F(0)),                qn(               F(2)) - qn(               F(0)),
				    qn(Q.rows()     + F(1)) - qn(Q.rows()     + F(0)), qn(Q.rows()     + F(2)) - qn(Q.rows()     + F(0)),
				    qn(Q.rows() * 2 + F(1)) - qn(Q.rows() * 2 + F(0)), qn(Q.rows() * 2 + F(2)) - qn(Q.rows() * 2 + F(0)),

			// Multiply on right by Xg inverse to get F
			XfXg *= DmInvList[i];

			if (XfXg.determinant() < 0) {
				std::cout << "detF < 0 !!!" << std::endl;
			}
			JacobiSVD<Matrix<double, 3, 2>> svd(XfXg, ComputeFullV | ComputeFullU);
			VectorXd svals = svd.singularValues();
			Matrix<double, 3, 2> S;
			S.setZero();
			S(0, 0) = std::max(min_sval, std::min(max_sval, svals(0)));
			S(1, 1) = std::max(min_sval, std::min(max_sval, svals(1)));

			if (svd.matrixU().determinant() < 0) {
				std::cout << "U ERRRRROR: " << std::endl;
			} 
			
			if (svd.matrixV().determinant() < 0) {
				std::cout << "V ERRRRROR: " << std::endl;
			}
			Matrix<double, 3, 2> T = svd.matrixU() * S * svd.matrixV().transpose();
			Matrix<double, 6, 1> Bp;
			Bp << T(0, 0), T(0, 1), T(1, 0), T(1, 1), T(2, 0), T(2, 1);
			BpList[i] = Bp;
		}

		// Computing RHS for global solver
		VectorXd RHS = (M / (dt*dt))*sn;
		IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

		for (int i = 0; i < F.rows(); ++i) {
			RHS += wtri * SList[i] * AList[i].transpose() * BpList[i];
		}

		for (int i = 0; i < positionIndices.size(); ++i) {
			RHS += wpos * SPosList[i] * Q0.row(positionIndices[i]);
		}
		//std::cout << "RHS : " << RHS.rows() << ", " << RHS.cols() << std::endl;
		//std::cout << RHS.format(CleanFmt) << std::endl;

		qn = solver.solve(RHS);
	}

	qdot = (qn.transpose() - q) / dt;
	q = qn.transpose();

	if (clickedFace != -1) {
		fExt(F(clickedFace, 0), 2) = 0;
		fExt(F(clickedFace, 1), 2) = 0;
		fExt(F(clickedFace, 2), 2) = 0;
	}

    return false;
}
