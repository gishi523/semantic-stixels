#ifndef __SEMANTIC_STIXELS_H__
#define __SEMANTIC_STIXELS_H__

#include <opencv2/core.hpp>

/** @brief Stixel geometric class id
*/
enum
{
	GEOMETRIC_ID_GROUND = 0,
	GEOMETRIC_ID_OBJECT = 1,
	GEOMETRIC_ID_SKY = 2,
};

/** @brief Stixel struct
*/
struct Stixel
{
	int uL;                   //!< stixel left x position
	int vT;                   //!< stixel top y position
	int vB;                   //!< stixel bottom y position
	int width;                //!< stixel width
	int geoId;                //!< stixel geometric class id
	int semId;                //!< stixel semantic class id
	float disp;               //!< stixel disparity
};

/** @brief CameraParameters struct
*/
struct CameraParameters
{
	float fu;                 //!< focal length x (pixel)
	float fv;                 //!< focal length y (pixel)
	float u0;                 //!< principal point x (pixel)
	float v0;                 //!< principal point y (pixel)
	float baseline;           //!< baseline (meter)
	float height;             //!< height position (meter)
	float tilt;               //!< tilt angle (radian)

	// default settings
	CameraParameters()
	{
		fu = 1.f;
		fv = 1.f;
		u0 = 0.f;
		v0 = 0.f;
		baseline = 0.2f;
		height = 1.f;
		tilt = 0.f;
	}
};

/** @brief SemanticStixels class.

The class implements the Semantic Stixel computation based on [1][2].
[1] Schneider, Lukas, et al. "Semantic stixels: Depth is not enough." 2016 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2016.
[2] Cordts, Marius, et al. "The stixel world: A medium-level representation of traffic scenes." Image and Vision Computing 68 (2017): 40-52.
*/
class SemanticStixels
{
public:

	enum
	{
		STIXEL_WIDTH_4 = 4, //!< stixel width
		STIXEL_WIDTH_8 = 8  //!< stixel width
	};

	enum
	{
		STIXEL_Y_RESOLUTION_4 = 4, //!< stixel vertical resolution
		STIXEL_Y_RESOLUTION_8 = 8  //!< stixel vertical resolution
	};

	enum
	{
		ROAD_ESTIMATION_AUTO = 0, //!< road disparity are estimated by input disparity
		ROAD_ESTIMATION_CAMERA    //!< road disparity are estimated by camera tilt and height
	};

	/** @brief Parameters struct
	*/
	struct Parameters
	{
		// disparity range
		int dmin;
		int dmax;

		// stixel width
		int stixelWidth;

		// stixel vertical resolution
		int stixelYResolution;

		// road disparity estimation
		int roadEstimation;

		// camera parameters
		CameraParameters camera;

		// geometry id for each class
		std::vector<int> geometry;

		// default settings
		Parameters()
		{
			// disparity range
			dmin = 0;
			dmax = 64;

			// stixel width
			stixelWidth = STIXEL_WIDTH_4;

			// stixel vertical resolution
			stixelYResolution = STIXEL_Y_RESOLUTION_4;

			// road disparity estimation
			roadEstimation = ROAD_ESTIMATION_AUTO;

			// camera parameters
			camera = CameraParameters();
		}
	};

	/** @brief Creates an instance of SemanticStixels.
		@param param Input parameters.
	*/
	static cv::Ptr<SemanticStixels> create(const Parameters& param = Parameters());

	/** @brief Computes semantic stixels from a disparity map and a disparity confidence.
		@param disparity Input 32-bit disparity map.
		@param stixels Output array of stixels.
	*/
	virtual void compute(const cv::Mat& disparity, std::vector<Stixel>& stixels) = 0;

	/** @brief Computes semantic stixels from a disparity map and a disparity confidence.
		@param disparity Input 32-bit disparity map.
		@param predict Input 32-bit 3-dimensional semantic segmentation scores.
		@param stixels Output array of stixels.
	*/
	virtual void compute(const cv::Mat& disparity, const cv::Mat& predict, std::vector<Stixel>& stixels) = 0;

	/** @brief Sets parameters to SemanticStixels.
		@param param Input parameters.
	*/
	virtual void setParameters(const Parameters& param) = 0;
};

#endif // !__SEMANTIC_STIXELS_H__
