#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace std;

namespace crapcode {
    float** loadWeights(char *path, int m, int n)
    {
        float **arr = new float* [m];
        if(arr == nullptr)
        {
            return arr;
        }

        for(int i = 0; i < m; i++)
        {
            arr[i] = new float[n];
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

    float* loadBiases(char *path, int n)
    {
        float *arr = new float[n];
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

    inline float relu(float x) 
    { 
        return x > 0 ? x : 0; 
    }

    void softmax(const float* input, int size, float* output) 
    {
        float max_val = input[0];
        for (int i = 1; i < size; i++) 
        {
            if (input[i] > max_val) 
            {
                max_val = input[i];
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < size; i++) 
        {
            output[i] = exp(input[i] - max_val);
            sum += output[i];
        }
        for (int i = 0; i < size; i++) 
        {
            output[i] /= sum;
        }
    }

    int forward(const float* input) {
        float **W1 = loadWeights("data/fc1.weight.txt", 784, 128);
        float *b1 = loadBiases("data/fc1.bias.txt", 128);
        if(W1 == nullptr | b1 == nullptr)
        {
            SDL_Log("Nullptr!");
            return 0;
        }
        
        float **W2 = loadWeights("data/fc2.weight.txt", 128, 64);
        float *b2 = loadBiases("data/fc2.bias.txt", 64);
        if(W2 == nullptr | b2 == nullptr)
        {
            SDL_Log("Nullptr!");
            return 0;
        }

        float **W3 = loadWeights("data/fc3.weight.txt", 64, 10);
        float *b3 = loadBiases("data/fc3.bias.txt", 10);
        if(W3 == nullptr | b3 == nullptr)
        {
            SDL_Log("Nullptr!");
            return 0;
        }

        if(input == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return -1;
        }

        float h1[128];
        for (int j = 0; j < 128; j++) 
        {
            float sum = b1[j];
            for (int i = 0; i < 784; i++) 
            {
                sum += input[i] * W1[i][j];
            }
            h1[j] = relu(sum);
        }

        float h2[64];
        for (int j = 0; j < 64; j++) 
        {
            float sum = b2[j];
            for (int i = 0; i < 128; i++) 
            {
                sum += h1[i] * W2[i][j];
            }
            h2[j] = relu(sum);
        }

        float out[10];
        for (int j = 0; j < 10; j++) 
        {
            float sum = b3[j];
            for (int i = 0; i < 64; i++)
            { 
                sum += h2[i] * W3[i][j];
            }
            out[j] = sum;
        }

        float probs[10];
        softmax(out, 10, probs);

        int pred = 0;
        SDL_Log("Predicition:");
        for (int i = 1; i < 10; i++) 
        {
            SDL_Log("%i - %f%\t", i, probs[i]*100);
            if (probs[i] > probs[pred]) 
            {
                pred = i;
            }
        }
        return pred;
    }
}
class Square
{
    int x, y, pixel, borderSize, w, h;
    float *data;
    public:
    Square(int sX = 0, int sY = 0, int pixSize = 5, int sBorderSize = 0, int width = 0, int height = 0) : x(sX), y(sY), pixel(pixSize), borderSize(sBorderSize), w(width), h(height)
    {
        data = new float[w * h];
        if(data == NULL)
        {
            SDL_Log("Не удалось создать массив");
            return;
        }
    }
    void fill()
    {
        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }

        for(int i = 0; i < w * h; i++)
        {
            data[i] = 1;
        }
    
    }
    void changePos(int newX = 0, int newY = 0)
    {
        x = newX;
        y = newY;
    }
    void draw(SDL_Renderer *rend = nullptr)
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
            SDL_FRect rect = {x - pixel * (i % w - (w - 2) / 2.0), y - pixel * (i / w - (h - 2) / 2.0), pixel, pixel };
            SDL_RenderFillRect(rend, &rect);
        }
    }
    void click(double mouseX = 0, double mouseY = 0, bool paint = true)
    {
        if(data == NULL)
        {
            SDL_Log("Нулевой указатель на массив 0_0");
            return;
        }
        int m = (x - mouseX + (w / 2.0) * pixel) / pixel;
        int n = (y - mouseY + (h / 2.0) * pixel) / pixel;
        if(m >= 0 & m < w)
        {
            if(n >= 0 & n < h)
            {
                if(paint)
                {
                    double distX = x - pixel * (m - (w - 2) / 2.0 - 0.5) - mouseX;
                    double distY = y - pixel * (n - (h - 2) / 2.0 - 0.5) - mouseY;
                    distX /= pixel / 2;
                    distY /= pixel / 2;
                    double dist = 1 - max(abs(distY), abs(distY));
                    if(dist > data[m + n * w])
                    {
                        data[m + n * w] = dist;
                    }
                }
                else
                {
                    data[m + n * w] = 0.0;
                }
            }
            crapcode::forward(data);
        }  
    }
    
    void destroy()
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
    Square sq = {400, 300, 10, 10, 28, 28};
    public:
    bool init()
    {
        char title[] = "Шлюшны Інтылект";
        return Program::init(title, 800, 600, SDL_WINDOW_RESIZABLE);
        sq.fill();
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
        sq.changePos(w / 2, h / 2);
        sq.draw(renderer);
        

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
                //SDL_Log("FPS: %i", frames);
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
                    if(e.button.button == 1) // Если левый клик - рисовать
                    {
                        sq.click(e.motion.x, e.motion.y, true);
                    }
                    else // Остальные клики - стирать
                    {
                        sq.click(e.motion.x, e.motion.y, false);
                    }
                }
                else if(e.type == SDL_EVENT_MOUSE_BUTTON_DOWN | e.type == SDL_EVENT_MOUSE_BUTTON_UP)
                {
                    hold = !hold;
                    if(e.button.button == 1) // Если левый клик - рисовать
                    {
                        sq.click(e.motion.x, e.motion.y, true);
                    }
                    else // Остальные клики - стирать
                    {
                        sq.click(e.motion.x, e.motion.y, false);
                    }
                }
            }
        }
        
    }
    void quit()
    {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        sq.destroy();
    }
};
int main()
{
    Intellect prog;
    prog.init();
    prog.loop();
    return 0;
}