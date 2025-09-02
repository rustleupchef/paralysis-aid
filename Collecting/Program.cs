using Newtonsoft.Json;

namespace Collecting;

using System.Threading;

class Program
{
    async Task Main()
    {
        dynamic json = JsonConvert.DeserializeObject(File.ReadAllText("config.json"));
        int times = json.times;
        string className = json.className;
        string path = json.path;
        
        Directory.CreateDirectory(Path.Combine(path, className));
        
        const string url = "http://localhost:3000/mindwave/data";
        HttpClient client = new HttpClient();
        client.BaseAddress = new Uri(url);

        Object[] largeJson = new Object[times];
        
        for (int i = 0; i < times; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                Console.WriteLine("Get ready in " + (5-j));
                Thread.Sleep(1000);
            }

            HttpResponseMessage response = await client.GetAsync(url);
            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine("This one didn't work..");
                break;
            }
            string responseString = await response.Content.ReadAsStringAsync();
            dynamic data = JsonConvert.DeserializeObject(responseString);
            string jsonEeg = JsonConvert.SerializeObject((string)data["eeg"]);
            largeJson[i] = jsonEeg;
        }
        string serializedJson = JsonConvert.SerializeObject(largeJson);
        File.WriteAllText(Path.Combine(path, className, $"{className}.json"), serializedJson);
    }
}