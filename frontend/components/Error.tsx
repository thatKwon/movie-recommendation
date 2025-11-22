type Props = {
  message: string;
};

export default function Error({ message }: Props) {
  return (
    <div className="flex justify-center items-center h-screen">
      <div className="text-red-500 text-2xl">{message}</div>
    </div>
  );
}
