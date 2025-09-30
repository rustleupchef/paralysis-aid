using Newtonsoft.Json;

namespace Collecting;

using System.Threading;

class Program
{
    async static Task Main()
    {
        dynamic json = JsonConvert.DeserializeObject(File.ReadAllText("config.json"));
        int times = json.times;
        int rest = json.rest;
        string className = json.className;
        string path = json.path;
        bool isMerge = json.isMerge;

        if (isMerge)
        {
            await mergeClasses(path);
            return;
        }

        Directory.CreateDirectory(Path.Combine(path, className));
        
        const string url = "http://localhost:3000/mindwave/data";
        HttpClient client = new HttpClient();
        client.BaseAddress = new Uri(url);

        Object[] largeJson = new Object[times];
        
        for (int i = 0; i < times; i++)
        {
            for (int j = 0; j < rest; j++)
            {
                Console.WriteLine("Get ready in " + (rest-j));
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
            largeJson[i] = data["eeg"];
            Console.WriteLine(data["eeg"]);
            Console.WriteLine("Completed");
        }
        string serializedJson = JsonConvert.SerializeObject(largeJson, Formatting.Indented);
        File.WriteAllText(Path.Combine(path, className, $"{className}.json"), serializedJson);
    }

    async static Task mergeClasses(string path)
    {
        string[] folders = Directory.GetDirectories(path);
        List<string> files = new List<string>();
        for (int i = 0; i < folders.Length; i++)
        {
            if (folders[i].Substring(folders[i].LastIndexOf("/") + 1) != "Formatted")
                files.Add(Path.Combine(path, folders[i],
                    folders[i].Substring(folders[i].LastIndexOf("/") + 1) + ".json"));
        }

        List<Object> objects = new List<Object>();
        for (int _label = 0; _label < files.Count; _label++)
        {
            dynamic json = JsonConvert.DeserializeObject(File.ReadAllText(files[_label]));

            foreach (dynamic jsonObject in json)
            {
                var payload = new
                {
                    features = new Object[] {jsonObject.delta, jsonObject.theta, jsonObject.loAlpha, jsonObject.hiAlpha, jsonObject.loBeta, jsonObject.hiBeta, jsonObject.loGamma, jsonObject.midGamma},
                    label = _label
                };
                objects.Add(payload);
            }
        }

        string text = JsonConvert.SerializeObject(objects, Formatting.Indented);
        File.WriteAllText(Path.Combine(path, "Formatted", $"Formatted.json"), text);

        Dictionary<string, string> key = new Dictionary<string, string>();
        for (int label = 0; label < files.Count; label++)
        {
            key.Add(label.ToString(), files[label].Substring(files[label].LastIndexOf("/") + 1, (files[label].Length - files[label].LastIndexOf("/")) - 6));
        }
        text = JsonConvert.SerializeObject(key, Formatting.Indented);
        File.WriteAllText(Path.Combine(path, "Formatted", $"ClassKey.json"), text);
    }
}