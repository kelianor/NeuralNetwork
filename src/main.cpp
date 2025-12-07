#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

class Layer
{
private:
    int output;
    double **weights, *bias; // weights[out][in] bias[out]
    Layer *inputNeurons;
public:
    double *neurons; // Bad practice :(

    Layer(int out, double **w, double *b, Layer *inNeurons, double *n) : output(out), weights(w), bias(b), inputNeurons(inNeurons), neurons(n)
    {
        
    }

    Layer(int out, double **w, double *b, Layer *inNeurons) : Layer(out, w, b, inNeurons, new double[out])
    {

    }
    double relu(double x)
    {
        return x > 0 ? x : 0;
    }
    void calculate()
    {
        
        if(neurons == nullptr)
        {
            SDL_Log("Нулевой указатель на массив нейронов!");
            return;
        }

        if(inputNeurons == nullptr)
        {
            SDL_Log("Нулевой указатель на входящий слой!");
            return;
        }

        for(int i = 0; i < output; i++)
        {
            neurons[i] = bias[i];
            for(int j = 0; j < (*inputNeurons).output; j++)
            {
                neurons[i] += (*inputNeurons).neurons[j] * weights[i][j];
            }
            neurons[i] = relu(neurons[i]);
        }
    }
    void deleteNeruons()
    {
        delete[] neurons;
        neurons = nullptr;
    }
};

class NumNeuralNetwork
{
private:
    double **weights1, **weights2, **weights3, *bias1, *bias2, *bias3, output[10];
    
    double** loadWeights(char *path, int m, int n)
    {
        double **arr = new double* [m];
        if(arr == nullptr)
        {
            return arr;
        }

        for(int i = 0; i < m; i++)
        {
            arr[i] = new double[n];
            if(arr[i] == nullptr)
            {
                return arr;
            }
        }


        ifstream file(path);
        if(!file.is_open()) {
            SDL_Log("Не удалось открыть файл %s", path);
            return nullptr;
        }

        string line;
        int row = 0;
        while(getline(file, line) && row < m) {
            istringstream ss(line);
            for(int col = 0; col < n; col++) {
                ss >> arr[row][col];
            }
            row++;
        }
        file.close();
        return arr;
    }

    double* loadBiases(char *path, int n)
    {
        double *arr = new double[n];
        if(arr == nullptr)
        {
            return arr;
        }

        ifstream file(path);
        if(!file.is_open()) {
            SDL_Log("Не удалось открыть файл %s", path);
            return nullptr;
        }

        for(int i = 0; i < n; i++) {
            file >> arr[i];
        }
        file.close();
        return arr;
    }
public:
    NumNeuralNetwork()
    {
        // Hidden layer 1
        weights1 = loadWeights("data/fc1.weight.txt", 784, 128);
        bias1 = loadBiases("data/fc1.bias.txt", 128);
        if(weights1 == nullptr | bias1 == nullptr)
        {
            SDL_Log("Nullptr!");
            return;
        }
        
        // Hidden layer 2
        weights2 = loadWeights("data/fc2.weight.txt", 128, 64);
        bias2 = loadBiases("data/fc2.bias.txt", 64);
        if(weights2 == nullptr | bias2 == nullptr)
        {
            SDL_Log("Nullptr!");
            return;
        }

        // Output layer
        weights3 = loadWeights("data/fc3.weight.txt", 64, 10);
        bias3 = loadBiases("data/fc3.bias.txt", 10);
        if(weights3 == nullptr | bias3 == nullptr)
        {
            SDL_Log("Nullptr!");
            return;
        }
    }
    double* recognise(double *input)
    {
        Layer in {784, nullptr, nullptr, nullptr, input};
        Layer hidden1 {128, weights1, bias1, &in};
        Layer hidden2 {64, weights2, bias2, &hidden1};
        Layer out {10, weights3, bias3, &hidden2};

        hidden1.calculate();
        hidden2.calculate();
        out.calculate();

        // Softmax the result
        
        double max = out.neurons[0];
        for (int i = 1; i < 10; i++) 
        {
            if (out.neurons[i] > max) 
            {
                max = out.neurons[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < 10; i++) 
        {
            output[i] = exp(out.neurons[i] - max);
            sum += output[i];
        }
        for (int i = 0; i < 10; i++) 
        {
            output[i] /= sum;
        }

        hidden1.deleteNeruons();
        hidden2.deleteNeruons();
        out.deleteNeruons();

        return output;
    }
};

class Rectangle
{
    int x, y, pixel, borderSize, w, h;
    public:
    double *data;
    Rectangle(int sX = 0, int sY = 0, int pixSize = 5, int sBorderSize = 0, int width = 0, int height = 0) : x(sX), y(sY), pixel(pixSize), borderSize(sBorderSize), w(width), h(height)
    {
        data = new double[w * h];
        if(data == NULL)
        {
            SDL_Log("Не удалось создать массив");
            return;
        }
    }
    void fill(double num = 0.0)
    {
        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }

        for(int i = 0; i < w * h; i++)
        {
            data[i] = num;
        }
    
    }
    void changePos(int newX = 0, int newY = 0)
    {
        x = newX;
        y = newY;
    }
    void render(SDL_Renderer *rend = nullptr)
    {
        // Border
        SDL_FRect rect = {x - pixel * w / 2.0 - borderSize, y - pixel * h / 2.0 - borderSize, pixel * w + borderSize * 2, pixel * h + borderSize * 2};
        if(!rend)
        {
            SDL_Log("Сцены не сущетсвует 0_0. Нулевой указатель на сцену");
            return;
        }
        SDL_RenderFillRect(rend, &rect);

        // Square content

        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }

        for(int i = 0; i < w * h; i++)
        {
            SDL_SetRenderDrawColor(rend, 255 * data[i], 255 * data[i], 255 * data[i], 0xFF);
            SDL_FRect rect = {x + pixel * (w / -2.0 + i % w), y + pixel * (h / -2.0 + i / w), pixel, pixel };
            SDL_RenderFillRect(rend, &rect);
        }
    }
    void clickAt(double mouseX = 0, double mouseY = 0, bool paint = true)
    {
        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }
        int m = (mouseX - x) / pixel + w / 2.0;
        int n = (mouseY - y) / pixel + h / 2.0;
        if(m >= 0 & m < w)
        {
            if(n >= 0 & n < h)
            {
                if(paint)
                {
                    data[m + n * w] = 1.0;
                }
                else
                {
                    data[m + n * w] = 0.0;
                }
            }
        }  
    }
    void quit()
    {
        delete[] data;
        data = nullptr;
    }
};
class Program
{
    protected:
    SDL_Window *window;
    SDL_Renderer *renderer;
    public:
    Program() : window(nullptr), renderer(nullptr) {};
    bool init(char *windowName, int w = 800, int h = 600, SDL_WindowFlags flags = 0)
    {
        if (!SDL_CreateWindowAndRenderer(windowName, w, h, flags, &window, &renderer)) 
        {
            SDL_Log("Не удалось создать окно: %s", SDL_GetError());
            return false;
        }

        SDL_Log("Используется рендеринг %s", SDL_GetRendererName(renderer));

        // Включение вертикальной синхронизации
        const int size = 4;
        int preffVsync[size] = {SDL_RENDERER_VSYNC_ADAPTIVE, 1, 2, SDL_RENDERER_VSYNC_DISABLED};
        bool vsync = false;
    
        for(int i = 0; i < size & !vsync; i++) // Выполнятется до тех пор пока не включится один из режимов вертикальной синхронизации
        {
            SDL_Log("Пробуем режим вертикальной синхронизации под номером %i", preffVsync[i]);
            vsync = SDL_SetRenderVSync(renderer, preffVsync[i]);
        }
        return true;
    }
    void quit()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};
class Intellect : public Program
{
    private:
    Rectangle paint = {400, 300, 15, 15, 28, 28};
    NumNeuralNetwork nw;
    double *results = nullptr;
    public:
    bool init()
    {
        paint.fill(0.0);
        results = nw.recognise(paint.data);
        char title[] = "Шлюшны Інтылект";
        return Program::init(title, 1200, 750, SDL_WINDOW_RESIZABLE);
    }
    void render()
    {
        // Black background
        SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
        SDL_RenderClear(renderer);

        // Rectangle

        SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0xAA, 0xFF);
        int w, h;
        SDL_GetWindowSizeInPixels(window, &w, &h);
        paint.changePos(w / 3, h / 2);
        paint.render(renderer);
        if(results != nullptr)
        {
            float scale = 3.0;
            float gap = 15.0*28.0 / 10.0 / scale;
            SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
            SDL_SetRenderScale(renderer, scale, scale);
            for(int i = 0; i < 10; i++)
            {
                string s = to_string(i) + " - " + to_string(results[i] * 100.0) + "%";
                SDL_RenderDebugText(renderer, (w / scale) * 2.0 / 3.0, (h / scale) / 2.0 - gap * 5.0 + gap * i , s.c_str());
            }
            SDL_SetRenderScale(renderer, 1.0, 1.0);
        }

        SDL_RenderPresent(renderer);
    }
    void loop()
    {
        bool hold = false;
        int frames = 0;
        SDL_Time lastTime = SDL_GetTicks();
        while(true)
        {
            render();

            // Calculating FPS
            SDL_Time currTime = SDL_GetTicks();
            SDL_Time deltaTime = currTime - lastTime;
            frames++;
            if(deltaTime > 1000)
            {
                lastTime = currTime;
                frames = 0;
            }


            SDL_Event e;
            while(SDL_PollEvent(&e))
            {
                if(e.type == SDL_EVENT_QUIT)
                {
                    quit();
                    return;
                } 
                else if(e.type == SDL_EVENT_MOUSE_MOTION & hold)
                {
                    paint.clickAt(e.motion.x, e.motion.y, e.button.button == 1);
                    results = nw.recognise(paint.data);
                }
                else if(e.type == SDL_EVENT_MOUSE_BUTTON_DOWN | e.type == SDL_EVENT_MOUSE_BUTTON_UP)
                {
                    hold = !hold;
                    paint.clickAt(e.motion.x, e.motion.y, e.button.button == 1);
                    results = nw.recognise(paint.data);
                }
            }
        }
        
    }
    void quit()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        paint.quit();
    }
};
int main()
{
    Intellect prog;
    prog.init();
    prog.loop();
    return 0;
}