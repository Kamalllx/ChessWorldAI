const Container = (props: any) => {
  return (
    <div className="d-flex h-100" style={{
      background: "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)",
      minHeight: "100vh"
    }}>
      {props.children}
    </div>
  )
}

export default Container;