import java.io.UnsupportedEncodingException;
import com.sun.deploy.net.URLEncoder;
import java.net.URLDecoder;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws UnsupportedEncodingException {
        String sentence = URLEncoder.encode("艺龙网并购两家旅游网站", "UTF-8");
        String url = "http://localhost:5000/online_test/";

        //发送 GET 请求
        String s = HttpUtil.sendGet(url, sentence);
        System.out.println("请求结果：" + s);
    }
}
