/*!

=========================================================
* Argon Dashboard React - v1.2.1
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-dashboard-react
* Copyright 2021 Creative Tim (https://www.creative-tim.com)
* Licensed under MIT (https://github.com/creativetimofficial/argon-dashboard-react/blob/master/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/

import {useState} from "react";
import {useDispatch} from "react-redux";

// reactstrap components
import {
    Alert,
    Button,
    Card,
    CardBody,
    Col,
    Form,
    FormGroup,
    Input,
    InputGroup,
    InputGroupAddon,
    InputGroupText
} from "reactstrap";

import {registerUser} from "../../_actions/user_actions";


const Register = ({history}) => {

    const [Username, setUsername] = useState("");
    const [Password, setPassword] = useState("");
    const [ConfirmPassword, setConfirmPassword] = useState("");

    const [CheckUsername, setCheckUsername] = useState(false);
    const [CheckPassword, setCheckPassword] = useState(false);

    const dispatch = useDispatch();

    const onChangeUsername = (e) => {
        setUsername(e.currentTarget.value);
    }

    const onChangePassword = (e) => {
        setPassword(e.currentTarget.value);
    }

    const onChangeConfirmPassword = (e) => {
        setConfirmPassword(e.currentTarget.value);
    }

    const onSubmitHandler = (e) => {

        e.preventDefault();

        if (Username && Password && Password === ConfirmPassword) {

            const user = {
                username: Username,
                password: Password
            };

            setCheckUsername(false);
            setCheckPassword(false);

            dispatch(registerUser(user))
                .then((res) => {
                    if (res.payload.success) {
                        history.push("/auth/login");
                    } else {
                        alert('Fail to Register');
                    }
                })
                .catch((err) => {
                    // status: 500
                    // statusText: "Internal Server Error"
                    // -> 기가입된 Username 을 입력하면 발생
                    console.log(err);
                    setCheckUsername(true);
                })
        } else if (Username) {
            setCheckUsername(false);
            setCheckPassword(true);
        } else if (CheckPassword) {
            setCheckUsername(true);
            setCheckPassword(false);
        } else {
            setCheckUsername(true);
            setCheckPassword(true);
        }
    }

    return (
        <>
            <Col lg="6" md="8">
                <Card className="bg-secondary shadow border-0">
                    <CardBody className="px-lg-5 py-lg-5">
                        <Form role="form" onSubmit={onSubmitHandler}>
                            {
                                CheckUsername &&
                                <Alert color="danger">
                                    <span className="alert-inner--icon">
                                        <i className="ni ni-check-bold"/>
                                    </span>{" "}
                                    <span className="alert-inner--text">
                                        This username isn't allowed. Try again.
                                    </span>
                                </Alert>
                            }
                            <FormGroup>
                                <InputGroup className="input-group-alternative mb-3">
                                    <InputGroupAddon addonType="prepend">
                                        <InputGroupText>
                                            <i className="ni ni-circle-08"/>
                                        </InputGroupText>
                                    </InputGroupAddon>
                                    <Input placeholder="Username" type="text" value={Username}
                                           onChange={onChangeUsername}/>
                                </InputGroup>
                            </FormGroup>
                            {
                                CheckPassword &&
                                <Alert color="danger">
                                    <span className="alert-inner--icon">
                                        <i className="ni ni-check-bold"/>
                                    </span>{" "}
                                    <span className="alert-inner--text">
                                        Those passwords didn't match. Try again.
                                    </span>
                                </Alert>
                            }
                            <FormGroup>
                                <InputGroup className="input-group-alternative">
                                    <InputGroupAddon addonType="prepend">
                                        <InputGroupText>
                                            <i className="ni ni-lock-circle-open"/>
                                        </InputGroupText>
                                    </InputGroupAddon>
                                    <Input
                                        placeholder="Password"
                                        type="password"
                                        autoComplete="new-password"
                                        value={Password}
                                        onChange={onChangePassword}
                                    />
                                </InputGroup>
                            </FormGroup>
                            <FormGroup>
                                <InputGroup className="input-group-alternative">
                                    <InputGroupAddon addonType="prepend">
                                        <InputGroupText>
                                            <i className="ni ni-lock-circle-open"/>
                                        </InputGroupText>
                                    </InputGroupAddon>
                                    <Input
                                        placeholder="Confirm Password"
                                        type="password"
                                        autoComplete="new-password"
                                        value={ConfirmPassword}
                                        onChange={onChangeConfirmPassword}
                                    />
                                </InputGroup>
                            </FormGroup>
                            <div className="text-center">
                                <Button className="mt-4" color="primary" type="button" onClick={onSubmitHandler}>
                                    Create account
                                </Button>
                            </div>
                        </Form>
                    </CardBody>
                </Card>
            </Col>
        </>
    );
};

export default Register;
