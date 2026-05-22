#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// ---------------------------------------------------------
// 1. FILTRU MEDIAN 5x5
// ---------------------------------------------------------
Mat filtruMedianOptimizat(const Mat& src) {
    Mat dst = Mat::zeros(src.size(), src.type());
    int offset = 2; // fereastra 5x5

#pragma omp parallel for
    for (int i = offset; i < src.rows - offset; i++) {
        for (int j = offset; j < src.cols - offset; j++) {


            uchar valori[25];
            int index = 0;

            for (int u = -offset; u <= offset; u++) {
                for (int v = -offset; v <= offset; v++) {
                    valori[index++] = src.at<uchar>(i + u, j + v);
                }
            }

            // Sortează array-ul și ia valoarea din mijloc
            sort(valori, valori + 25);
            dst.at<uchar>(i, j) = valori[12];
        }
    }
    return dst;
}

// ---------------------------------------------------------
// 2. FILTRU GAUSSIAN 5x5 OPTIMIZAT
// ---------------------------------------------------------
Mat filtruGaussianOptimizat(const Mat& src) {
    const int kernel[5][5] = {
        {1,  4,  7,  4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1,  4,  7,  4, 1}
    };
    const int c = 273;
    Mat dst = Mat::zeros(src.size(), src.type());
    int offset = 2;

#pragma omp parallel for
    for (int i = offset; i < src.rows - offset; i++) {
        for (int j = offset; j < src.cols - offset; j++) {
            int suma = 0;
            for (int u = 0; u < 5; u++) {
                for (int v = 0; v < 5; v++) {
                    suma += kernel[u][v] * src.at<uchar>(i + u - offset, j + v - offset);
                }
            }
            dst.at<uchar>(i, j) = static_cast<uchar>(suma / c);
        }
    }
    return dst;
}

// ---------------------------------------------------------
// 3. CALCUL DEPTH MAP (SAD - Sum of Absolute Differences)
// ---------------------------------------------------------
Mat calculeazaDepthMapSAD(const Mat& stanga, const Mat& dreapta, int fereastra, int max_disparitate) {
    Mat depthMap = Mat::zeros(stanga.size(), CV_8UC1);
    int offset = fereastra / 2;

#pragma omp parallel for
    for (int y = offset; y < stanga.rows - offset; y++) {
        for (int x = offset + max_disparitate; x < stanga.cols - offset; x++) {

            int min_SAD = INT_MAX;
            int best_disparitate = 0;

            for (int d = 0; d < max_disparitate; d++) {
                int sad_curent = 0;

                for (int wy = -offset; wy <= offset; wy++) {
                    for (int wx = -offset; wx <= offset; wx++) {
                        uchar pixel_stanga = stanga.at<uchar>(y + wy, x + wx);
                        uchar pixel_dreapta = dreapta.at<uchar>(y + wy, x + wx - d);
                        sad_curent += abs(pixel_stanga - pixel_dreapta);
                    }
                }

                if (sad_curent < min_SAD) {
                    min_SAD = sad_curent;
                    best_disparitate = d;
                }
            }

            depthMap.at<uchar>(y, x) = static_cast<uchar>((best_disparitate * 255) / max_disparitate);
        }
    }
    return depthMap;
}

// ---------------------------------------------------------
// 4. ANAGLIFĂ SINTETICĂ (Modulată fix de Depth Map)
// ---------------------------------------------------------
Mat creeazaAnaglifaCuDepthCorect(const Mat& stanga, const Mat& dreapta, const Mat& depthMap, int max_disparitate) {
    Mat anaglifa = Mat::zeros(stanga.size(), CV_8UC3);
    float factor_confort = 0.1f;

#pragma omp parallel for
    for (int y = 0; y < stanga.rows; y++) {
        for (int x = 0; x < stanga.cols; x++) {

            // Ochiul Stâng Vede Canalul ROȘU (Nemodificat)
            anaglifa.at<Vec3b>(y, x)[2] = stanga.at<uchar>(y, x);

            // Ochiul Drept vede Canalele VERDE și ALBASTRU (Cyan).
            // Le mutăm bazându-ne pe Harta de Adâncime!
            int disparitate_pixel = (depthMap.at<uchar>(y, x) * max_disparitate * factor_confort) / 255; 
            int x_ochi_drept = x - disparitate_pixel;

            if (x_ochi_drept >= 0) {
                uchar cyan = dreapta.at<uchar>(y, x_ochi_drept);
                anaglifa.at<Vec3b>(y, x)[1] = cyan; // Verde
                anaglifa.at<Vec3b>(y, x)[0] = cyan; // Albastru
            }
            else {
                anaglifa.at<Vec3b>(y, x)[1] = 0;
                anaglifa.at<Vec3b>(y, x)[0] = 0;
            }
        }
    }
    return anaglifa;
}

int main() {
    // ---------------------------------------------------------
    // A. MENIU INTERACTIV
    // ---------------------------------------------------------
    int optiune;
    string cale_stanga, cale_dreapta, nume_video;

    cout << "==========================================\n";
    cout << "   PROCESARE VIDEO 3D (STEREO ANAGLIF)    \n";
    cout << "==========================================\n";
    cout << "Alege secventa de test:\n";
    cout << " 1 - Secventa Test 1\n";
    cout << " 2 - Secventa Test 2\n";
    cout << " 3 - Secventa Test 3\n";
    cout << " 4 - Secventa Test 4\n";
    cout << " 5 - Secventa Test 5\n";
    cout << "==========================================\n";
    cout << "Introdu numarul (1-5): ";

    if (!(cin >> optiune)) {
        cout << "Eroare: Nu ai introdus un numar valid!" << endl;
        return -1;
    }

    switch (optiune) {
    case 1:
        cale_stanga = "test/test_spiderman/frames_L";
        cale_dreapta = "test/test_spiderman/frames_R";
        nume_video = "film_anaglif_3D_test1.avi";
        break;
    case 2:
        cale_stanga = "test/test_rollercoaster/frames_L";
        cale_dreapta = "test/test_rollercoaster/frames_R";
        nume_video = "film_anaglif_3D_test2.avi";
        break;
    case 3:
        cale_stanga = "test/test_ball/frames_L";
        cale_dreapta = "test/test_ball/frames_R";
        nume_video = "film_anaglif_3D_test3.avi";
        break;
    case 4:
        cale_stanga = "test/test_helicopter/frames_L";
        cale_dreapta = "test/test_helicopter/frames_R";
        nume_video = "film_anaglif_3D_test4.avi";
        break;
    case 5:
        cale_stanga = "test/test_bunny/frames_L";
        cale_dreapta = "test/test_bunny/frames_R";
        nume_video = "film_anaglif_3D_test5.avi";
        break;        
    default:
        cout << "Optiune invalida!" << endl;
        return -1;
    }

    cout << "\nSe proceseaza:\n- " << cale_stanga << "\n- " << cale_dreapta << "\n\n";

    // ---------------------------------------------------------
    // B. CITIREA ȘI SORTAREA FIȘIERELOR
    // ---------------------------------------------------------
    vector<string> fisiere_stanga, fisiere_dreapta;

    if (!fs::exists(cale_stanga) || !fs::exists(cale_dreapta)) {
        cout << "Eroare: Folderele nu exista pe disc!" << endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(cale_stanga))
        if (entry.path().extension() == ".bmp") fisiere_stanga.push_back(entry.path().string());

    for (const auto& entry : fs::directory_iterator(cale_dreapta))
        if (entry.path().extension() == ".bmp") fisiere_dreapta.push_back(entry.path().string());

    sort(fisiere_stanga.begin(), fisiere_stanga.end());
    sort(fisiere_dreapta.begin(), fisiere_dreapta.end());

    if (fisiere_stanga.empty() || fisiere_stanga.size() != fisiere_dreapta.size()) {
        cout << "Eroare: Numar diferit de imagini sau foldere goale!" << endl;
        return -1;
    }

    // ---------------------------------------------------------
    // C. CONFIGURARE IEȘIRE VIDEO
    // ---------------------------------------------------------
    Mat tempImg = imread(fisiere_stanga[0], IMREAD_GRAYSCALE);
    Size dimCadru = tempImg.size();

    int fps = 15;
    VideoWriter videoFinal(nume_video, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, dimCadru, true);

    // ---------------------------------------------------------
    // D. PROCESAREA CADRU CU CADRU
    // ---------------------------------------------------------
    int fereastra_cautare = 11;
    int max_disparitate = 32;

    for (size_t i = 0; i < fisiere_stanga.size(); i++) {
        Mat imgStanga = imread(fisiere_stanga[i], IMREAD_GRAYSCALE);
        Mat imgDreapta = imread(fisiere_dreapta[i], IMREAD_GRAYSCALE);

        if (imgStanga.empty() || imgDreapta.empty()) continue;

        // 1. Calculăm Harta de Adâncime Manuală
        Mat depthMap = calculeazaDepthMapSAD(imgStanga, imgDreapta, fereastra_cautare, max_disparitate);

        // 2. Aplicăm Post-Procesare (Simulăm un SGBM "sărac")
        depthMap = filtruMedianOptimizat(depthMap);
        depthMap = filtruMedianOptimizat(depthMap);
        depthMap = filtruGaussianOptimizat(depthMap);

        // 3. Creăm Anaglifa folosind Harta Curățată
        Mat anaglifa = creeazaAnaglifaCuDepthCorect(imgStanga, imgDreapta, depthMap, max_disparitate);

        // 4. Salvare și afișare
        videoFinal.write(anaglifa);
        imshow("Harta Adancime", depthMap);
        imshow("Anaglifa (Sintetica 3D)", anaglifa);

        cout << "Cadru procesat: " << i + 1 << "/" << fisiere_stanga.size() << endl;

        if (waitKey(1) == 27) break; // ESC pentru oprire
    }

    videoFinal.release();
    destroyAllWindows();

    cout << "\nSUCCES! Video salvat ca '" << nume_video << "'." << endl;
    return 0;
}