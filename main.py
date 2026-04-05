'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-04 10:53:50
 * @Description  : 
'''
from config import configure_settings, load_config
from service import RagService


def main():
    # 读/写/持久化各自分离
    config = load_config()
    configure_settings(config)
    rag_service = RagService(config)

    try:
        while True:
            question = input("\nQuestion (q to quit): ").strip()
            if question.lower() == "q":
                break
            if not question:
                continue

            result = rag_service.query(question=question)
            print("\n" + "=" * 50)
            print(result["answer"])
            print("\nSources:")
            for index, source in enumerate(result["sources"], start=1):
                print(f"\n[{index}] {source['source_file']} score={source['score']:.3f}")
                print(source["text"])
            print("=" * 50)
    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    main()
