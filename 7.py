import os
import subprocess
import sys


def main():
    print("=" * 60)
    print("动物零样本识别系统启动器")
    print("=" * 60)
    print()

    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")

    if not os.path.exists(app_path):
        print(f"错误：找不到 app.py 文件")
        print(f"请将本脚本与 app.py 放在同一目录下")
        input("按回车键退出...")
        return

    print(f"正在启动 Streamlit 应用...")
    print(f"应用路径: {app_path}")
    print()

    # 启动 Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        input("按回车键退出...")


if __name__ == "__main__":
    main()