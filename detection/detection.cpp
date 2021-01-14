/*  
    First of all, we would like to thank our teacher-Ms Lan. Who helped us a lot during this project.
    This program is inspired by electric cars from Tesla. 
    It could help detect lanes which is an important feature of self-driving cars.
    We tried our best to implement this idea. However, it isn't really perfect.
    We'll so happy to receive any suggestions.
    Thanks for reading.
*/

/*
-------steps as below-------
_______1.RGB to gray level
_______2.noise reduction
_______3.edge detection-  implemented by Le Viet Duc	 // 20192777 
_______4.line detection-  implemented by Ngo Thi Thu Hoa // 20192859
*/
#include <iostream>
#include<cmath>
#include< iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#define thre1 70 // threshold in edge
#define myTHRESHOLD 90
#define num_lines 200
using namespace std;
using namespace cv;
string dir_img =  "1.jpg" ;// path of image.

class laneDetection {
private:
	Mat img_input,
		gray_img,
		blur_img,
		edge_img,
		line_img;
public:
	laneDetection(string  dir) 
	{
		this->img_input = imread(dir);
        resize(img_input,img_input, Size(480, 360));
	}
	int xGradient(Mat image, int x, int y);
	int yGradient(Mat image, int x, int y);
	Mat edgeDetection();
	void lineDetection();
    void show_img();
	void plot();
};
void laneDetection::show_img()
{
    imshow("gray", gray_img);
    imshow("blur",  blur_img);
    imshow("edge", edge_img);
   // imshow("line", line_img);
    waitKey();
}
int laneDetection::xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +       //giai thich uchar
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}
 
int laneDetection::yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y-1, x) +
                 image.at<uchar>(y-1, x+1) -
                  image.at<uchar>(y+1, x-1) -
                   2*image.at<uchar>(y+1, x) -
                    image.at<uchar>(y+1, x+1);
}

Mat laneDetection::edgeDetection()
{	/* Here, we used functions available in opencv */
    cvtColor(img_input, gray_img, COLOR_BGR2GRAY);// convert RGB_image to grayscale
    GaussianBlur(gray_img, blur_img, Size(9, 9), 2);//convert gray_image  to blur_image

    // Declare variables in Canny 
    Mat src, dst,dst2;
    int gx, gy, sum, g;
    double theta;
    // Load an image
    src = blur_img;
    dst = src.clone();// for gradient of each pixel
    dst2 = src.clone();// for theta of each pixel

    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            dst.at<uchar>(y, x) = 0.0;   //creat an new array "0", which is same size as blur_img.
            dst2.at<uchar>(y, x) = 0.0;
        }

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            gx = xGradient(src, x, y); 
            gy = yGradient(src, x, y);
            g = sqrt(gx * gx + gy * gy); // compute gradient.
            if (gx == 0)gx = 1;         // if gx==0, this program would fail.
            /* compute theta */
            theta = atan(gy / gx);        //cout << gx<<"     "<<gy<<endl;                     
            theta = theta * 180 / 3.1415;
            //cout << theta << endl;

            /*Rounding theta  angle to the values: 0, 45, 90, 135. */
            if (theta >= -22 && theta <= 22)
                theta = 0;
            else if (theta >= 154)
                theta = 0;
            else if (theta > 22 && theta <= 68)

                theta = 45;
            else if (theta > -154 && theta <= -110)

                theta = 45;
            else if (theta > 68 && theta < 112)
                theta = 90;
            else if (theta > -110 && theta < -66)
                theta = 90;
            else if (theta >= -66 && theta <= -22)theta = 135;
            else theta = 135;
            //cout << "tick:"<<theta << endl;
            /* Now, we have 2 variables of Mat type.
            dst represents gradient and dst2 represents theta angle */
            dst.at<uchar>(y, x) = g;
            dst2.at<uchar>(y, x) = theta;
        }
    }
    /* lọc các điểm khả năng cao là biên  */
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
           g= dst.at<uchar>(y, x);
           theta= dst2.at<uchar>(y, x);
          
           switch(int(theta))
           {
           case 0:
               if ((g > dst.at<uchar>(y, x + 1)) && (g > dst.at<uchar>(y, x - 1)))
                    continue;
               else  dst.at<uchar>(y, x) = 0;
               break;

           case 45:
               if ((g > dst.at<uchar>(y - 1, x + 1)) && (g > dst.at<uchar>(y + 1, x - 1)))
                   continue;
               else  dst.at<uchar>(y, x) = 0;
               break;
           case 90:
               if ((g > dst.at<uchar>(y-1, x )) && (g > dst.at<uchar>(y+1, x )))
                   continue;
               else  dst.at<uchar>(y, x) = 0;
               break;
           case 135:
               if ((g > dst.at<uchar>(y+1, x + 1)) && (g > dst.at<uchar>(y-1, x - 1)))
                   continue;
               else  dst.at<uchar>(y, x) = 0;
               break;
           }
       
           
        }
    }
    /* Threshold */
    for (int y = 1; y < src.rows - 1; y++) 
    {
        for (int x = 1; x < src.cols - 1; x++) 
        {   
            if (y<100)
                dst.at<uchar>(y, x) = 0;
            if (dst.at<uchar>(y, x) >= thre1)
                dst.at<uchar>(y, x) = 255;
            else dst.at<uchar>(y, x) = 0;
        }
    }
    this->edge_img = dst;
        /* preview images */
        //namedWindow("edge");
        //imshow("blur", blur_img);
        //imshow("edge", edge_img);
        /* export images to use in slide */
        //imwrite("blur_img.jpg", blur_img);
        //imwrite("gray_img.jpg",gray_img);
        //imwrite("input.jpg", img_input);
        //imwrite("edge_img.jpg", dst);
        //waitKey(0);
        return edge_img;
    }

void laneDetection::lineDetection()
{
    this->line_img = img_input.clone();
    double rho = 1;             //sThe rho has a value between [-diagonal_length, diagonal_length]
    double theta = 3.14 / 180;// The theta angle has a value between 0 and 180
    int threshold = 100;
    int rows = 2*sqrt(edge_img.rows * edge_img.rows + edge_img.cols * edge_img.cols)+1;// diagonal length = 600*2+1
    int cols = 180; //180 degree
    int  *Accumulator=new int[rows*cols];//declare an array to count same values of rho anh theta
    
    //Initialize array "0"
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            Accumulator[i * cols + j] = 0;
    /*Count the number of white pixels*/
    int count_255=0;
    for (int y = 1; y < edge_img.rows - 1; y++) 
    {
        for (int x = 1; x < edge_img.cols - 1; x++)
        {
            if (edge_img.at<uchar>(y, x) == 255)
            {
                count_255++;
            }
        }
    }
    //cout << "255" << count_255<<endl;
    double *matrix1 =new double [2*count_255];	// store coordinates
    double *matrix2 =new double [2*180];		// store values : cos,sin 
    int *matrix3 =new int [count_255*180]; // matrix3= matrix1*matrix2
    /*find coordinate of white pixels.*/ 
    int count_pixel=0;
    for (int y = 1; y < edge_img.rows - 1; y++) 
    {
        for (int x = 1; x < edge_img.cols - 1; x++)
        {
            if (edge_img.at<uchar>(y, x) == 255)
            {	
                matrix1[count_pixel*2]=double(x);
                matrix1[count_pixel*2+1]= double(y);
                count_pixel++;
                //cout << count_pixel << endl;
                //cout << x << "-" << y << endl;
            }
        }
    }
    /*coumpute sin,cos from -90 to 90 degree*/
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 180; j++)
           {
           	if(i==0)
           	{	double t=double(j-90);
           		matrix2[i*180+j]=cos(t*(3.14/180));
           		//cout<< matrix2[i * 1 + j] <<" ";
           	}
           	if(i==1)
           	{	double t=double(j-90);
           		matrix2[i*180+j]=sin(t*(3.14/180));
           	}
            //cout << matrix2[i * 1 + j] << " ";
           	//cout<<endl;
           }
   /*compute rho*/
    for(int i=0;i<count_255;i++)  
	{  
		for(int j=0;j<180;j++)  
		{  
			matrix3[i*180+j]=0;  
			for(int k=0;k<2;k++)  
			{  
			matrix3[i*180+j] += matrix1[i*2+k]*matrix2[k*180+j];   
            //cout << matrix1[i * 2 + k] << "  " << matrix2[k * 180 + j] << endl;
            }  
            //cout << "rho:"<<matrix3[i * 180 + j] << endl;
          // 	cout<<j<<" ";
		}  
		//cout<<endl<<i<<endl;
	}  

    /*accumulate*/
	for(int i=0;i<count_255;i++)  
	{  
		for(int j=0;j<180;j++) 
		{   
            int temp = matrix3[i * 180 + j];
            temp += 600;// convert negative values to positive
            
            Accumulator[temp*180+j]++;
            //cout << temp << " " << endl;
		}
      //  cout << endl;
	}

    /*choose lines*/
    
    int *lines = new int[num_lines*2]; // matrix (2*10) store lines.row 1 - rho,row2 -theta.
    for (int i = 0; i < 2 * num_lines; i++)
        lines[i] = 0;
    int count_line = 0;
    for (int i = 0; i < rows; i++)//rho
    {
        for (int j = 0; j < cols; j++)//theta
        {   
            if (Accumulator[i * cols + j] >= myTHRESHOLD)// we assigned it is 100
            {
                for (int k = 0; k < 2; k++) 
                {   
                if(k==0)
                    lines[count_line *  2 + k] = i;//rho
                if(k==1)
                    lines[count_line *  2 + k] = j;//theta
                }
                count_line++;
            }

        }
        //cout << endl;
    }
    cout << "num_lines: " << count_line << endl;
    cout << setw(10) << "rho";
    cout << setw(10) << "Theta";
    cout << endl;

    /* draw lines*/
    for (int line_i = 0; line_i < count_line; line_i++)
    {
        int rho_f = lines[line_i * 2];
        int theta_f = lines[line_i * 2 + 1];
        cout <<"theta: "<< theta_f << endl;
        // cout <<setw(10)<< rho << setw(10)<< theta << endl;
        int x_f = 0, a1 = 0, b1 = 0;//a1,b1 -> min 
        int y_f = 0, a2 = 0, b2 = 0;//a2,b2 -> max
        /*travle matrix3, find rho, find x-y max and min of a line*/
        int temp_ab = 0;
        for (int i3 = 0; i3 < count_255; i3++)//on matrix3
        {
            for (int j3 = 0; j3 < 180; j3++)
            {

                // cout << setw(10) << (matrix3[i3 * 180 + j3 ]+600)<< setw(10) << j3 << endl;
                if ((matrix3[i3 * 180 + j3] + 600 == rho_f) && (j3 == theta_f))
                {
                    x_f = matrix1[i3 * 2];
                    y_f = matrix1[i3 * 2 + 1];
                    
                    //cout << "X" << "-Y" << endl;
                    //cout << x_f << " " << y_f<<endl;
                    if (temp_ab == 0) {
                        a1 = x_f;  b1 = y_f;
                    
                    temp_ab++;
                    }
                }
                
                //cout<<"y_f:" << y_f<<" ";
                if (x_f < a1)
                {
                    a1 = x_f; b1 = y_f;
                }
                if (x_f > a2)
                {
                    a2 = x_f; b2 = y_f;
                }
            }
        }
        cout << "--------------" << endl;
        cout << "a1:" << a1 << " b1:" << b1 << endl;
        cout << "a2:" << a2 << " b2:" << b2 << endl<<endl;
        Point start(a1, b1);
        Point end(a2, b2);
        /*draw line to original image*/
        
        line(line_img, start, end, Scalar(0,255, 0), 2, LINE_8);
        

    }
    //for (int i = 0; i < rows; i++)
    //    for (int j = 0; j < cols; j++)
           // cout << Accumulator[i * cols + j] << endl;
    /* Display images in steps*/
    imshow("input", img_input);
    imshow("gray", gray_img);
    imshow("blur", blur_img);
    imshow("edge", edge_img);
    imshow("line", line_img);
    imwrite("input.jpg", img_input);
    imwrite("blur.jpg", blur_img);
    imwrite("edge.jpg", edge_img);
    imwrite("line.jpg", line_img);
    waitKey(0);
    delete []Accumulator;
    delete []matrix3;
    delete []matrix1;
    delete []matrix2;
    delete []lines;
}
//void laneDetection::plot(){	}

int main()
{
   
    laneDetection img = laneDetection(dir_img);
	Mat for_line= img.edgeDetection();
    img.lineDetection();
    //img.show_img();
	return 0;
}

